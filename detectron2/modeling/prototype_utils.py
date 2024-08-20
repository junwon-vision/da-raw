import torch
import math
import torch.nn as nn
import torch.nn.functional as F

from detectron2.utils.comm import all_gather_diff_size
from .sinkhorn import distributed_sinkhorn
from .info_nce import InfoNCELoss
import torch.distributed as dist


# class-agnostic prototype loss
def prototype_loss(contrast_features, scores, prototypes, score_threshold):
    # normalize prototypes
    with torch.no_grad():
        w = prototypes.data.clone()
        w = nn.functional.normalize(w, dim=1, p=2)
        prototypes.copy_(w)
    
    contrast_features = torch.cat(contrast_features)
    scores = torch.cat(scores)
    
    # get only features over scores
    scores = scores[:,:-1]
    max_score, cls = scores.max(dim=1)
    
    fg_index = max_score > score_threshold
    
    contrast_features = contrast_features[fg_index]

    contrast_features = nn.functional.normalize(contrast_features, dim=1, p=2)
    similarity = torch.matmul(contrast_features, prototypes.clone().t()) # n*p
    
    # get assignments
    with torch.no_grad():
        sim = similarity.detach()
        assign_matrix = distributed_sinkhorn(sim) # n*p
    
    if len(assign_matrix) == 0:
        proto_loss = torch.tensor(0.0).to(torch.device('cuda'))
    else:
        proto_loss = -torch.mean(torch.sum(assign_matrix * F.log_softmax(similarity/0.1, dim=1), dim=1)) * 0.1

    return proto_loss



# per class prototype
def per_class_prototype_loss(contrast_features, scores, prototypes, score_threshold, 
                             use_bg = True,
                             num_class = 8,
                             num_prototype = 5,
                             ):
    
    # normalize prototypes
    with torch.no_grad():
        w = prototypes.data.clone()
        w = nn.functional.normalize(w, dim=1, p=2)
        prototypes.copy_(w)
    
    contrast_features = torch.cat(contrast_features)
    scores = torch.cat(scores)
    
    # only get features over score threshold
    max_score, cls = scores.max(dim=1)
    keep_over_th = max_score > score_threshold
    keep_over_bg_th = max_score > 0.9
    
    # include features classified as background if needed
    fg_keep = torch.logical_and(keep_over_th,cls < num_class)
    bg_keep = torch.logical_and(keep_over_bg_th, cls==num_class)
    if use_bg:
        keep = torch.logical_or(fg_keep, bg_keep)    
    else:
        keep = fg_keep
    
    contrast_features = contrast_features[keep]
    cls = cls[keep]
    max_score = max_score[keep].detach()
    
    # get similarity btw contrast features & prototypes
    contrast_features = nn.functional.normalize(contrast_features, dim=1, p=2)
    similarity = torch.matmul(contrast_features, prototypes.clone().t()) # n*(p*c)
    
    proto_cont_loss = torch.tensor(0.0).to(torch.device('cuda'))

    # For each class, use sinkhorn-knopp algorithm to get assignment.
    # Use the assignment for CE Loss
    
    # gather all instances to use distributed_sinkhorn
    all_similarity = all_gather_diff_size(similarity)
    all_cls = torch.cat([cls.cuda() for cls in all_gather_diff_size(cls)])
    len_similarity = [len(sim) for sim in all_similarity]
    all_similarity = torch.cat([sim.cuda() for sim in all_similarity])
    
    cls_assign = torch.zeros_like(all_similarity).cuda()
    
    for current_cls in range(num_class+use_bg):
        if (all_cls == current_cls).sum()==0: continue
        
        with torch.no_grad():
            all_cls_proto_sim = all_similarity[all_cls==current_cls,num_prototype*current_cls:num_prototype*(current_cls+1)]
            cls_proto_sim = all_cls_proto_sim.detach()
            cls_proto_assign = distributed_sinkhorn(cls_proto_sim) # n*p
            cls_assign[all_cls==current_cls,num_prototype*current_cls:num_prototype*(current_cls+1)] = cls_proto_assign
    
    if keep.sum() == 0:
        return proto_cont_loss
    
    # For multi-gpu
    rank = dist.get_rank() if len(len_similarity) > 1 else 0
    
    my_list = [0]
    for i in range(len(len_similarity)):
        num = sum(len_similarity[:i+1])
        my_list.append(num)
    
    start, end = my_list[rank], my_list[rank+1]
    gpu_cls_assign = cls_assign[start:end]
    
    # bg prototype loss
    proto_cont_loss += -torch.mean(torch.sum(gpu_cls_assign * F.log_softmax(similarity/0.1, dim=1), dim=1))
    proto_cont_loss /= (num_class+use_bg)
    
    return proto_cont_loss
    
    
    
# contrastive loss. make features from same class closer, different class further?
def contrastive_loss(box_features, scores, 
                     INST_SCORE_THRESH = 0.7,
                     temperature = 0.1,
                     ):
    
    dim, num_cls = scores.shape
    max_score, cls = scores.max(dim=1)
    
    loss = torch.tensor(0.0).to(torch.device('cuda'))
    
    # For pseudo-label, use only foreground and high score instances
    keep_over_th = max_score > INST_SCORE_THRESH
    keep_fg = cls < (num_cls-1)
    keep = torch.logical_and(keep_over_th,keep_fg)
    
    keep_over_bg_th = max_score > 0.9

    bg_keep = torch.logical_and(keep_over_bg_th, cls==(num_cls-1))
    bg_feats = box_features[bg_keep]
    
    if keep.sum() == 0:
        return loss
    
    cls = cls[keep]
    box_features = box_features[keep] # N*1024
    
    infoNCE = InfoNCELoss(temperature)
    
    for i in range(len(box_features)):
        query = box_features[i]
        
        pos_mask = (cls == cls[i])
        neg_mask = torch.logical_not(pos_mask)
        pos_mask[i] = False
        
        positive_keys = box_features[pos_mask]
        negative_keys = box_features[neg_mask]
        negative_keys = torch.cat((negative_keys, bg_feats))
        
        loss += infoNCE(query, positive_keys, negative_keys)
    
    loss /= len(box_features)
    return loss
