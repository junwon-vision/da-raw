import os
import torch
import torch.nn as nn

def domain_loss_img(img_feat, target=False, reduction='mean'):
    # DA IMG LOSS
    # up_sample = nn.Upsample(size=(H, W), mode='bilinear', align_corners=True)
    upsampled_loss = []
    BCEloss = nn.BCELoss(reduction=reduction)
    for i, feat, in enumerate(img_feat):
        
        feat_label = torch.zeros_like(feat, dtype=torch.float32)
        if target: feat_label[None,:] = 1
        level_loss = BCEloss(feat, feat_label)
        upsampled_loss.append(level_loss)
    da_img_loss = torch.stack(upsampled_loss)
    da_img_loss = da_img_loss.mean()
    
    da_inst_loss = torch.zeros(1).cuda()

    da_cons_loss = torch.zeros(1).cuda()
    return da_img_loss, da_inst_loss, da_cons_loss


def domain_loss_img_singlelayer(img_feat, target=False, reduction='mean'):
    # DA IMG LOSS
    BCEloss = nn.BCELoss(reduction=reduction)

    feat_label = torch.zeros_like(img_feat, dtype=torch.float32)
    if target: feat_label[None,:] = 1
    level_loss = BCEloss(img_feat, feat_label)
    
    return level_loss

class FocalLoss(nn.Module):
    def __init__(self, gamma=5):
        super(FocalLoss, self).__init__()
        self.gamma = gamma
        
    def forward(self, input, target):
        """
        input: [N, ], float32
        target: [N, ], int64
        """
        
        src_term = (1-input)**self.gamma *torch.log(input)
        tgt_term = (input)**self.gamma * torch.log(1-input)
        
        loss = src_term*target + tgt_term*(1-target)
        return -loss.mean()

class DALoss(object):
    """
    Domain Adaptive Loss
    """
    def __init__(self, cfg):
        self.cfg = cfg.clone()
        # in_features       = cfg_model.ROI_HEADS.IN_FEATURES
        pooler_resolution = cfg.MODEL.ROI_BOX_HEAD.POOLER_RESOLUTION

        self.avgpool = nn.AvgPool2d(kernel_size=pooler_resolution, stride=pooler_resolution)
        self.l1_loss = nn.L1Loss()
        
    def __call__(self, 
                 img_feat=None, inst_feat=None, tgt_inst_index=None,
                 cons_img_feat=None, cons_inst_feat=None, 
                 proposals=None, cons_box_pooler=None, 
                 target=False, cons_loss = True,
                 adapt_level=None):
        
        da_img_loss = torch.zeros(1)
        da_inst_loss = torch.zeros(1)
        da_cons_loss = torch.zeros(1)
        # 
        # DA IMG LOSS
        upsampled_loss = []
        BCEloss = nn.BCELoss(reduction='mean')
        Focalloss = FocalLoss()
        # 
        if adapt_level in ['image', 'imginst', 'all']:
            half_bs = int(len(img_feat[0])/2)
            for i, feat, in enumerate(img_feat) :
                feat_label = torch.zeros_like(feat, dtype=torch.float32)
                feat_label[half_bs:] = 1
                if feat.dim() == 2:
                    level_loss = Focalloss(feat, feat_label)
                else:
                    level_loss = BCEloss(feat, feat_label)
                upsampled_loss.append(level_loss)
                
            da_img_loss = torch.stack(upsampled_loss)
            da_img_loss = da_img_loss.mean()
        # 
        # DA INST LOSS
        if adapt_level in ['inst', 'imginst', 'all']:
            inst_feat_cat = torch.cat(inst_feat)
            cons_inst_feat_cat = torch.cat(cons_inst_feat)
            inst_label = torch.zeros_like(inst_feat_cat, dtype=torch.float32)
            
            tgt_index = torch.cat(tgt_inst_index)
            inst_label[tgt_index] = 1
            # if target: inst_label[None,:] = 1
            da_inst_loss = BCEloss(inst_feat_cat, inst_label)

        # DA CONS LOSS
        if adapt_level == 'all':
            box_lists = [x.proposal_boxes for x in proposals]
            da_img_rois_probs, index_list = cons_box_pooler(cons_img_feat, box_lists)
            da_img_rois_probs_pool = self.avgpool(da_img_rois_probs)
            
            cons_img_feat_per_level = []
            for i, inds in enumerate(index_list):
                cons_img_feat_per_level.append(da_img_rois_probs_pool[inds].squeeze(3).squeeze(2).squeeze(1))

            cons_img_feat_cat = torch.cat(cons_img_feat_per_level)
            # da_img_rois_probs_pool = da_img_rois_probs_pool.view(da_img_rois_probs_pool.size(0), -1)
            
            da_cons_loss = self.l1_loss(cons_img_feat_cat, cons_inst_feat_cat)
                
        return da_img_loss, da_inst_loss, da_cons_loss
