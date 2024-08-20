# Copyright (c) Facebook, Inc. and its affiliates.
import logging
import numpy as np
from typing import Dict, List, Optional, Tuple
import torch
import torch.nn.functional as F
from torch import nn
import cv2
import matplotlib.pyplot as plt
import torch.distributed as dist

from detectron2.config import configurable
from detectron2.data.detection_utils import convert_image_to_rgb
from detectron2.layers import move_device_like
from detectron2.structures import Boxes, ImageList, Instances
from detectron2.utils.events import get_event_storage
from detectron2.utils.logger import log_first_n
from detectron2.utils.comm import get_world_size, is_main_process, all_gather_diff_size


from ..backbone import Backbone, build_backbone
from ..postprocessing import detector_postprocess
from ..proposal_generator import build_proposal_generator
from ..roi_heads import build_roi_heads
from .build import META_ARCH_REGISTRY
from ..net_utils import AdaptiveReverseLayerF, ReverseLayerF, ScaleLayerF, domain_pixel_cls, domain_inst_cls, domain_img_cls, domain_img_cls_GRAM, domain_img_cls_GRAM_thin
from ..cbam_utils import CBAM
from ..losses import domain_loss_img, domain_loss_img_singlelayer, DALoss
import detectron2.utils.comm as comm
from ..prototype_utils import per_class_prototype_loss, contrastive_loss

__all__ = ["GeneralizedRCNN", "GeneralizedRCNN_DA", "ProposalNetwork"]


@META_ARCH_REGISTRY.register()
class GeneralizedRCNN(nn.Module):
    """
    Generalized R-CNN. Any models that contains the following three components:
    1. Per-image feature extraction (aka backbone)
    2. Region proposal generation
    3. Per-region feature extraction and prediction
    """

    @configurable
    def __init__(
        self,
        *,
        backbone: Backbone,
        proposal_generator: nn.Module,
        roi_heads: nn.Module,
        pixel_mean: Tuple[float],
        pixel_std: Tuple[float],
        input_format: Optional[str] = None,
        vis_period: int = 0,
    ):
        """
        Args:
            backbone: a backbone module, must follow detectron2's backbone interface
            proposal_generator: a module that generates proposals using backbone features
            roi_heads: a ROI head that performs per-region computation
            pixel_mean, pixel_std: list or tuple with #channels element, representing
                the per-channel mean and std to be used to normalize the input image
            input_format: describe the meaning of channels of input. Needed by visualization
            vis_period: the period to run visualization. Set to 0 to disable.
        """
        super().__init__()
        self.backbone = backbone
        self.proposal_generator = proposal_generator
        self.roi_heads = roi_heads

        self.input_format = input_format
        self.vis_period = vis_period
        if vis_period > 0:
            assert input_format is not None, "input_format is required for visualization!"

        self.register_buffer("pixel_mean", torch.tensor(pixel_mean).view(-1, 1, 1), False)
        self.register_buffer("pixel_std", torch.tensor(pixel_std).view(-1, 1, 1), False)
        assert (
            self.pixel_mean.shape == self.pixel_std.shape
        ), f"{self.pixel_mean} and {self.pixel_std} have different shapes!"

    @classmethod
    def from_config(cls, cfg):
        backbone = build_backbone(cfg)
        return {
            "backbone": backbone,
            "proposal_generator": build_proposal_generator(cfg, backbone.output_shape()),
            "roi_heads": build_roi_heads(cfg, backbone.output_shape()),
            "input_format": cfg.INPUT.FORMAT,
            "vis_period": cfg.VIS_PERIOD,
            "pixel_mean": cfg.MODEL.PIXEL_MEAN,
            "pixel_std": cfg.MODEL.PIXEL_STD,
        }

    @property
    def device(self):
        return self.pixel_mean.device

    def _move_to_current_device(self, x):
        return move_device_like(x, self.pixel_mean)

    def visualize_training(self, batched_inputs, proposals):
        """
        A function used to visualize images and proposals. It shows ground truth
        bounding boxes on the original image and up to 20 top-scoring predicted
        object proposals on the original image. Users can implement different
        visualization functions for different models.

        Args:
            batched_inputs (list): a list that contains input to the model.
            proposals (list): a list that contains predicted proposals. Both
                batched_inputs and proposals should have the same length.
        """
        from detectron2.utils.visualizer import Visualizer

        storage = get_event_storage()
        max_vis_prop = 20
        for input, prop in zip(batched_inputs, proposals):
            img = input["image"]
            img = convert_image_to_rgb(img.permute(1, 2, 0), self.input_format)
            v_gt = Visualizer(img, None)
            v_gt = v_gt.overlay_instances(boxes=input["instances"].gt_boxes)
            anno_img = v_gt.get_image()
            box_size = min(len(prop.proposal_boxes), max_vis_prop)
            v_pred = Visualizer(img, None)
            v_pred = v_pred.overlay_instances(
                boxes=prop.proposal_boxes[0:box_size].tensor.cpu().numpy()
            )
            prop_img = v_pred.get_image()
            vis_img = np.concatenate((anno_img, prop_img), axis=1)
            vis_img = vis_img.transpose(2, 0, 1)
            vis_name = "Left: GT bounding boxes;  Right: Predicted proposals"
            storage.put_image(vis_name, vis_img)
            break  # only visualize one image in a batch

    def forward(self, batched_inputs: List[Dict[str, torch.Tensor]]):
        """
        Args:
            batched_inputs: a list, batched outputs of :class:`DatasetMapper` .
                Each item in the list contains the inputs for one image.
                For now, each item in the list is a dict that contains:

                * image: Tensor, image in (C, H, W) format.
                * instances (optional): groundtruth :class:`Instances`
                * proposals (optional): :class:`Instances`, precomputed proposals.

                Other information that's included in the original dicts, such as:

                * "height", "width" (int): the output resolution of the model, used in inference.
                  See :meth:`postprocess` for details.

        Returns:
            list[dict]:
                Each dict is the output for one input image.
                The dict contains one key "instances" whose value is a :class:`Instances`.
                The :class:`Instances` object has the following keys:
                "pred_boxes", "pred_classes", "scores", "pred_masks", "pred_keypoints"
        """
        if not self.training:
            return self.inference(batched_inputs)
        
        images = self.preprocess_image(batched_inputs)
        if "instances" in batched_inputs[0]:
            gt_instances = [x["instances"].to(self.device) for x in batched_inputs]
        else:
            gt_instances = None
            
        
        features = self.backbone(images.tensor)
        # print(features['p2'])
        # 
        
        if self.proposal_generator is not None:
            proposals, proposal_losses = self.proposal_generator(images, features, gt_instances)
        else:
            assert "proposals" in batched_inputs[0]
            proposals = [x["proposals"].to(self.device) for x in batched_inputs]
            proposal_losses = {}

        _, detector_losses = self.roi_heads(images, features, proposals, gt_instances)
        
        if self.vis_period > 0:
            storage = get_event_storage()
            if storage.iter % self.vis_period == 0:
                self.visualize_training(batched_inputs, proposals)
        
        losses = {}
        losses.update(detector_losses)
        losses.update(proposal_losses)
        if losses['loss_cls'].data > 100:
            
        # print(losses)
        return losses

    def inference(
        self,
        batched_inputs: List[Dict[str, torch.Tensor]],
        detected_instances: Optional[List[Instances]] = None,
        do_postprocess: bool = True,
    ):
        """
        Run inference on the given inputs.

        Args:
            batched_inputs (list[dict]): same as in :meth:`forward`
            detected_instances (None or list[Instances]): if not None, it
                contains an `Instances` object per image. The `Instances`
                object contains "pred_boxes" and "pred_classes" which are
                known boxes in the image.
                The inference will then skip the detection of bounding boxes,
                and only predict other per-ROI outputs.
            do_postprocess (bool): whether to apply post-processing on the outputs.

        Returns:
            When do_postprocess=True, same as in :meth:`forward`.
            Otherwise, a list[Instances] containing raw network outputs.
        """
        assert not self.training

        images = self.preprocess_image(batched_inputs)
        features = self.backbone(images.tensor)

        if detected_instances is None:
            if self.proposal_generator is not None:
                proposals, _ = self.proposal_generator(images, features, None)
            else:
                assert "proposals" in batched_inputs[0]
                proposals = [x["proposals"].to(self.device) for x in batched_inputs]

            results, _ = self.roi_heads(images, features, proposals, None)
        else:
            detected_instances = [x.to(self.device) for x in detected_instances]
            results = self.roi_heads.forward_with_given_boxes(features, detected_instances)

        if do_postprocess:
            assert not torch.jit.is_scripting(), "Scripting is not supported for postprocess."
            return GeneralizedRCNN._postprocess(results, batched_inputs, images.image_sizes)
        return results

    def preprocess_image(self, batched_inputs: List[Dict[str, torch.Tensor]]):
        """
        Normalize, pad and batch the input images.
        """
        images = [self._move_to_current_device(x["image"]) for x in batched_inputs]
        images = [(x - self.pixel_mean) / self.pixel_std for x in images]
        images = ImageList.from_tensors(
            images,
            self.backbone.size_divisibility,
            padding_constraints=self.backbone.padding_constraints,
        )
        return images

    @staticmethod
    def _postprocess(instances, batched_inputs: List[Dict[str, torch.Tensor]], image_sizes):
        """
        Rescale the output instances to the target size.
        """
        # note: private function; subject to changes
        processed_results = []
        for results_per_image, input_per_image, image_size in zip(
            instances, batched_inputs, image_sizes
        ):
            height = input_per_image.get("height", image_size[0])
            width = input_per_image.get("width", image_size[1])
            # height = image_size[0]
            # width = image_size[1]
            # 
            r = detector_postprocess(results_per_image, height, width)
            processed_results.append({"instances": r})
        return processed_results


@META_ARCH_REGISTRY.register()
class ProposalNetwork(nn.Module):
    """
    A meta architecture that only predicts object proposals.
    """

    @configurable
    def __init__(
        self,
        *,
        backbone: Backbone,
        proposal_generator: nn.Module,
        pixel_mean: Tuple[float],
        pixel_std: Tuple[float],
    ):
        """
        Args:
            backbone: a backbone module, must follow detectron2's backbone interface
            proposal_generator: a module that generates proposals using backbone features
            pixel_mean, pixel_std: list or tuple with #channels element, representing
                the per-channel mean and std to be used to normalize the input image
        """
        super().__init__()
        self.backbone = backbone
        self.proposal_generator = proposal_generator
        self.register_buffer("pixel_mean", torch.tensor(pixel_mean).view(-1, 1, 1), False)
        self.register_buffer("pixel_std", torch.tensor(pixel_std).view(-1, 1, 1), False)

    @classmethod
    def from_config(cls, cfg):
        backbone = build_backbone(cfg)
        return {
            "backbone": backbone,
            "proposal_generator": build_proposal_generator(cfg, backbone.output_shape()),
            "pixel_mean": cfg.MODEL.PIXEL_MEAN,
            "pixel_std": cfg.MODEL.PIXEL_STD,
        }

    @property
    def device(self):
        return self.pixel_mean.device

    def _move_to_current_device(self, x):
        return move_device_like(x, self.pixel_mean)

    def forward(self, batched_inputs):
        """
        Args:
            Same as in :class:`GeneralizedRCNN.forward`

        Returns:
            list[dict]:
                Each dict is the output for one input image.
                The dict contains one key "proposals" whose value is a
                :class:`Instances` with keys "proposal_boxes" and "objectness_logits".
        """
        images = [self._move_to_current_device(x["image"]) for x in batched_inputs]
        images = [(x - self.pixel_mean) / self.pixel_std for x in images]
        images = ImageList.from_tensors(
            images,
            self.backbone.size_divisibility,
            padding_constraints=self.backbone.padding_constraints,
        )
        features = self.backbone(images.tensor)

        if "instances" in batched_inputs[0]:
            gt_instances = [x["instances"].to(self.device) for x in batched_inputs]
        elif "targets" in batched_inputs[0]:
            log_first_n(
                logging.WARN, "'targets' in the model inputs is now renamed to 'instances'!", n=10
            )
            gt_instances = [x["targets"].to(self.device) for x in batched_inputs]
        else:
            gt_instances = None
        proposals, proposal_losses = self.proposal_generator(images, features, gt_instances)
        # In training, the proposals are not useful at all but we generate them anyway.
        # This makes RPN-only models about 5% slower.
        if self.training:
            return proposal_losses

        processed_results = []
        for results_per_image, input_per_image, image_size in zip(
            proposals, batched_inputs, images.image_sizes
        ):
            height = input_per_image.get("height", image_size[0])
            width = input_per_image.get("width", image_size[1])
            r = detector_postprocess(results_per_image, height, width)
            processed_results.append({"proposals": r})
        return processed_results



@META_ARCH_REGISTRY.register()
class GeneralizedRCNN_DA(nn.Module):
    """
    Generalized R-CNN. Any models that contains the following three components:
    1. Per-image feature extraction (aka backbone)
    2. Region proposal generation
    3. Per-region feature extraction and prediction
    """

    @configurable
    def __init__(
        self,
        *,
        backbone: Backbone,
        proposal_generator: nn.Module,
        roi_heads: nn.Module,
        pixel_mean: Tuple[float],
        pixel_std: Tuple[float],
        input_format: Optional[str] = None,
        vis_period: int = 0,
        cfgs = None
    ):
        """
        Args:
            backbone: a backbone module, must follow detectron2's backbone interface
            proposal_generator: a module that generates proposals using backbone features
            roi_heads: a ROI head that performs per-region computation
            pixel_mean, pixel_std: list or tuple with #channels element, representing
                the per-channel mean and std to be used to normalize the input image
            input_format: describe the meaning of channels of input. Needed by visualization
            vis_period: the period to run visualization. Set to 0 to disable.
        """
        super().__init__()
        self.backbone = backbone
        self.proposal_generator = proposal_generator
        self.roi_heads = roi_heads

        self.input_format = input_format
        self.vis_period = vis_period
        if vis_period > 0:
            assert input_format is not None, "input_format is required for visualization!"

        self.register_buffer("pixel_mean", torch.tensor(pixel_mean).view(-1, 1, 1), False)
        self.register_buffer("pixel_std", torch.tensor(pixel_std).view(-1, 1, 1), False)
        assert (
            self.pixel_mean.shape == self.pixel_std.shape
        ), f"{self.pixel_mean} and {self.pixel_std} have different shapes!"
        self.output_dir = cfgs.OUTPUT_DIR
        # DA modules

        # GET PARAMETERS
        self.da_apply_layer = cfgs.MODEL.DA.DA_APPLY_LAYER
        self.cbam = cfgs.MODEL.DA.CBAM
        self.dout = 256
        self.dout_inst = 1024
        self.alpha1 = cfgs.MODEL.DA.ALPHA_1
        self.alpha2 = cfgs.MODEL.DA.ALPHA_2
        self.alpha_mid = cfgs.MODEL.DA.ALPHA_MID
        self.cons_weight = cfgs.MODEL.DA.COS_WEIGHT
        self.img_weight = cfgs.MODEL.DA.IMG_WEIGHT
        
        ## Contrastive Learning
        if cfgs.MODEL.CL.MODE is None:
            self.contrast = False
            
        elif cfgs.MODEL.CL.MODE == 'contrast':
            #instance contrastive learning
            self.contrast = cfgs.MODEL.CL.MODE
            self.contrast_weight = cfgs.MODEL.CL.CONTRAST_WEIGHT
            self.temp = cfgs.MODEL.CL.CONTRAST_TEMP
            self.contrast_score_threshold = cfgs.MODEL.CL.CONTRAST_SCORE_THRESHOLD

        elif cfgs.MODEL.CL.MODE == 'per_class_prototype':
            ## PROTOTYPE BASED CLUSTERING
            self.contrast = cfgs.MODEL.CL.MODE
            self.num_prototype = cfgs.MODEL.CL.NUM_PROTOTYPE
            self.contrast_score_threshold = cfgs.MODEL.CL.CONTRAST_SCORE_THRESHOLD
            self.use_source = cfgs.MODEL.CL.USE_SOURCE
            self.num_class = cfgs.MODEL.ROI_HEADS.NUM_CLASSES
            self.contrast_weight = cfgs.MODEL.CL.CONTRAST_WEIGHT
            in_channels = 128
            
            use_bg = True
            if use_bg:
                self.offset = 1
            else:
                self.offset = 0
                
            self.prototypes = nn.Parameter(torch.zeros(self.num_prototype*(self.num_class+self.offset), in_channels),
                                        requires_grad = True)
            nn.init.trunc_normal_(self.prototypes, std=0.02)
            
            self.score_att = cfgs.MODEL.CL.SCORE_ATT
            self.contrast_layer = cfgs.MODEL.CL.CONTRAST_LAYER
        
        ## image level
        self.mid_features = cfgs.MODEL.DA.GET_MID_FEATURES
        self.mid_da_apply_layer = cfgs.MODEL.DA.MID_DA_APPLY_LAYER
        
        self.adapt_level = cfgs.MODEL.DA.ADAPT_LEVEL
        
        self.gram = False
        self.cons_loss = True
        if cfgs.MODEL.DA.IMG_CLS == 'pixel':
            self.DA_img_p2 = domain_pixel_cls(self.dout)
            self.DA_img_p3 = domain_pixel_cls(self.dout)
            self.DA_img_p4 = domain_pixel_cls(self.dout)
            self.DA_img_p5 = domain_pixel_cls(self.dout)
            self.img_disc_list = {'p2': self.DA_img_p2, 'p3': self.DA_img_p3, 'p4': self.DA_img_p4, 'p5': self.DA_img_p5}
        elif cfgs.MODEL.DA.IMG_CLS == 'image':
            self.cons_loss = False
            self.DA_img_p2 = domain_img_cls(self.dout)
            self.DA_img_p3 = domain_img_cls(self.dout)
            self.DA_img_p4 = domain_img_cls(self.dout)
            self.DA_img_p5 = domain_img_cls(self.dout)
            self.img_disc_list = {'p2': self.DA_img_p2, 'p3': self.DA_img_p3, 'p4': self.DA_img_p4, 'p5': self.DA_img_p5}
        elif cfgs.MODEL.DA.IMG_CLS == 'SWDA':
            self.gram = True
            self.DA_img_p2 = domain_pixel_cls(self.dout)
            self.DA_img_p3 = domain_pixel_cls(self.dout)
            self.DA_img_p4 = domain_img_cls(self.dout)
            self.DA_img_p5 = domain_img_cls(self.dout)
            self.img_disc_list = {'p2': self.DA_img_p2, 'p3': self.DA_img_p3, 'p4': self.DA_img_p4, 'p5': self.DA_img_p5}
        else:
            assert(0)
        
        ## instance level
        if self.adapt_level in ['inst', 'imginst', 'all']:
            self.DA_inst_p2 = domain_inst_cls(self.dout_inst)
            self.DA_inst_p3 = domain_inst_cls(self.dout_inst)
            self.DA_inst_p4 = domain_inst_cls(self.dout_inst)
            self.DA_inst_p5 = domain_inst_cls(self.dout_inst)
            self.inst_disc_list = {'p2': self.DA_inst_p2, 'p3': self.DA_inst_p3, 'p4': self.DA_inst_p4, 'p5': self.DA_inst_p5}
        self.DAloss = DALoss(cfgs)
        
        
        ## CBAM Layer 
        if self.cbam:
            self.CBAM_p4 = CBAM(self.dout)
            self.CBAM_p5 = CBAM(self.dout)
            self.cbam_list = {'p4': self.CBAM_p4, 'p5': self.CBAM_p5}
        
    @classmethod
    def from_config(cls, cfg):
        backbone = build_backbone(cfg)
        return {
            "backbone": backbone,
            "proposal_generator": build_proposal_generator(cfg, backbone.output_shape()),
            "roi_heads": build_roi_heads(cfg, backbone.output_shape()),
            "input_format": cfg.INPUT.FORMAT,
            "vis_period": cfg.VIS_PERIOD,
            "pixel_mean": cfg.MODEL.PIXEL_MEAN,
            "pixel_std": cfg.MODEL.PIXEL_STD,
            "cfgs": cfg,
        }

    @property
    def device(self):
        return self.pixel_mean.device

    def _move_to_current_device(self, x):
        return move_device_like(x, self.pixel_mean)

    def visualize_training(self, batched_inputs, proposals):
        """
        A function used to visualize images and proposals. It shows ground truth
        bounding boxes on the original image and up to 20 top-scoring predicted
        object proposals on the original image. Users can implement different
        visualization functions for different models.

        Args:
            batched_inputs (list): a list that contains input to the model.
            proposals (list): a list that contains predicted proposals. Both
                batched_inputs and proposals should have the same length.
        """
        from detectron2.utils.visualizer import Visualizer

        storage = get_event_storage()
        max_vis_prop = 20
        dist = 50
        
        for input, prop in zip(batched_inputs, proposals):
            img = input["image"]
            img = convert_image_to_rgb(img.permute(1, 2, 0), self.input_format)
            v_gt = Visualizer(img, None)
            v_gt = v_gt.overlay_instances(boxes=input["instances"].gt_boxes)
            anno_img = v_gt.get_image()
            box_size = min(len(prop.proposal_boxes), max_vis_prop)
            # 
            v_pred = Visualizer(img, None)
            v_pred = v_pred.overlay_instances(
                boxes=prop.proposal_boxes[0:box_size].tensor.cpu().numpy()
            )
            prop_img = v_pred.get_image()

            vis_img = np.concatenate((anno_img, prop_img), axis=1)
            vis_img = vis_img.transpose(2, 0, 1)
            vis_name = "Left: GT bounding boxes;  Right: Predicted proposals"
            storage.put_image(vis_name, vis_img)
            break  # only visualize one image in a batch

    def forward(self, batched_inputs: List[Dict[str, torch.Tensor]], is_taodac = False, show_feature = False):
        """
        Args:
            batched_inputs: a list, batched outputs of :class:`DatasetMapper` .
                Each item in the list contains the inputs for one image.
                For now, each item in the list is a dict that contains:

                * image: Tensor, image in (C, H, W) format.
                * instances (optional): groundtruth :class:`Instances`
                * proposals (optional): :class:`Instances`, precomputed proposals.

                Other information that's included in the original dicts, such as:

                * "height", "width" (int): the output resolution of the model, used in inference.
                  See :meth:`postprocess` for details.

        Returns:
            list[dict]:
                Each dict is the output for one input image.
                The dict contains one key "instances" whose value is a :class:`Instances`.
                The :class:`Instances` object has the following keys:
                "pred_boxes", "pred_classes", "scores", "pred_masks", "pred_keypoints"
        """
        if not self.training:
            return self.inference(batched_inputs, show_feature = show_feature)
        
        
        
        # if taodac, change classes to person, car, motorcycle
        if is_taodac:
            for iter,input in enumerate(batched_inputs):
                gt_classes = input['instances'].get('gt_classes')
                taodac_classes = []
                for i in gt_classes:
                    if i in [0, 1]: taodac_classes.append(0)
                    elif i in [2, 3, 4, 5]: taodac_classes.append(2)
                    elif i in [6, 7]: taodac_classes.append(6)
                        
                input['instances'].set('gt_classes', torch.tensor(taodac_classes))
        
        for i, x in enumerate(batched_inputs):
            
            if i >= len(batched_inputs)//2:
                batched_inputs[i]["instances"] = batched_inputs[i]["instances"][:0]
                
        
        # PRE-PROCESS INPUT
        images = self.preprocess_image(batched_inputs)
        
        if "instances" in batched_inputs[0]:
            gt_instances = [x["instances"].to(self.device) for x in batched_inputs]
        else:
            gt_instances = None

        # BACKBONE
        features = self.backbone(images.tensor)
        
        # RPN
        if self.proposal_generator is not None:
            proposals, proposal_losses = self.proposal_generator(images, features, gt_instances, only_source = True)
        else:
            assert "proposals" in batched_inputs[0]
            proposals = [x["proposals"].to(self.device) for x in batched_inputs]
            proposal_losses = {}
            
        # RCN
        new_proposals, detector_losses, feat_per_level, tgt_inst_index, cons_box_pooler, contrast_features, scores=\
            self.roi_heads(images, features, proposals, gt_instances)
        
        # get discriminator layer output
        da_img_feats, cons_img_feats = self.get_da_feat(features, level='image', alpha = self.alpha1)
        # 
        da_inst_feats, cons_inst_feats = self.get_da_feat(feat_per_level, level='inst', alpha = self.alpha2)

        # get domain Losses
        da_img_loss, da_inst_loss, da_cons_loss = self.DAloss(da_img_feats, da_inst_feats, tgt_inst_index,
                                                              cons_img_feats, cons_inst_feats, new_proposals, cons_box_pooler, 
                                                            cons_loss = self.cons_loss, adapt_level = self.adapt_level)

        losses = {}
        losses.update(detector_losses)
        losses.update(proposal_losses)
        
        if losses['loss_cls'].data > 100:
            
        
        # Contrastive Loss
        if self.contrast == 'contrast':
            cont_loss = torch.tensor(0.0).to(torch.device('cuda'))
            
            for i in range(0,2):
                cont_loss += self.contrast_weight * self.contrastive_loss(contrast_features[1][i], scores[1][i], 
                                                                          self.contrast_score_threshold)
            losses.update({'cont_loss': cont_loss})

        elif self.contrast == 'per_class_prototype':
            
            proto_cont_loss = torch.tensor(0.0).to(torch.device('cuda'))
            
            max_idx = max(self.contrast_layer)+1
            min_idx = min(self.contrast_layer)
            
            tgt_cont_loss = per_class_prototype_loss(contrast_features[1][min_idx:max_idx], scores[1][min_idx:max_idx],
                                                     self.prototypes, self.contrast_score_threshold)

            proto_cont_loss += tgt_cont_loss
            
            if self.use_source:
                src_cont_loss = per_class_prototype_loss(contrast_features[0][min_idx:max_idx], scores[0][min_idx:max_idx],
                                                         self.prototypes, self.contrast_score_threshold)
                proto_cont_loss += src_cont_loss

            losses.update({'cont_loss': proto_cont_loss*self.contrast_weight})
            
        # only update needed losses
        if self.adapt_level == 'image':
            losses.update({'da_img_loss':da_img_loss*self.img_weight})
        elif self.adapt_level == 'inst':
            losses.update({'da_inst_loss':da_inst_loss})
        elif self.adapt_level == 'imginst':
            losses.update({'da_img_loss':da_img_loss, 'da_inst_loss':da_inst_loss})
        elif self.adapt_level == 'all':
            losses.update({'da_img_loss':da_img_loss, 'da_inst_loss':da_inst_loss, 'da_cons_loss':da_cons_loss})

        return losses

    
    def inference(
        self,
        batched_inputs: List[Dict[str, torch.Tensor]],
        detected_instances: Optional[List[Instances]] = None,
        do_postprocess: bool = True,
        show_feature = False
    ):
        """
        Run inference on the given inputs.

        Args:
            batched_inputs (list[dict]): same as in :meth:`forward`
            detected_instances (None or list[Instances]): if not None, it
                contains an `Instances` object per image. The `Instances`
                object contains "pred_boxes" and "pred_classes" which are
                known boxes in the image.
                The inference will then skip the detection of bounding boxes,
                and only predict other per-ROI outputs.
            do_postprocess (bool): whether to apply post-processing on the outputs.

        Returns:
            When do_postprocess=True, same as in :meth:`forward`.
            Otherwise, a list[Instances] containing raw network outputs.
        """
        assert not self.training
        # 
        images = self.preprocess_image(batched_inputs)
        features = self.backbone(images.tensor)
        
        if detected_instances is None:
            if self.proposal_generator is not None:
                proposals, _ = self.proposal_generator(images, features, None) # RPN
            else:
                assert "proposals" in batched_inputs[0]
                proposals = [x["proposals"].to(self.device) for x in batched_inputs]
            # results, _ = self.roi_heads(images, features, proposals, None)
            results, inst_feat = self.roi_heads(images, features, proposals, None)
        else:
            detected_instances = [x.to(self.device) for x in detected_instances]
            results = self.roi_heads.forward_with_given_boxes(features, detected_instances)

        if do_postprocess:
            assert not torch.jit.is_scripting(), "Scripting is not supported for postprocess."
            if show_feature:
                return GeneralizedRCNN_DA._postprocess(results, batched_inputs, images.image_sizes), inst_feat
            return GeneralizedRCNN_DA._postprocess(results, batched_inputs, images.image_sizes)
        if show_feature:
            return results, inst_feat
        return results
    
    def visualize_prototype(
        self,
        batched_inputs: List[Dict[str, torch.Tensor]],
        detected_instances: Optional[List[Instances]] = None,
        do_postprocess: bool = True,
        use_gt_proposals: bool = False,
    ):
        assert not self.training
        # 
        images = self.preprocess_image(batched_inputs)
        features = self.backbone(images.tensor)
        
        sp_att, ch_att = self.get_da_feat(features, level='image', alpha = self.alpha1, get_att = True)
        
        if not use_gt_proposals:
            proposals, _ = self.proposal_generator(images, features, None) # RPN
            results, inst_feat, contrast_feat, keep_indices = self.roi_heads.get_contrast_feat_and_results(features, proposals)
        else:
            assert "proposals" in batched_inputs[0]
            proposals = [x["proposals"].to(self.device) for x in batched_inputs]
            results, inst_feat, contrast_feat, keep_indices = self.roi_heads.get_contrast_feat_and_results(features, proposals, use_gt=True)
            
        prototypes = self.prototypes
        
        if do_postprocess:
            assert not torch.jit.is_scripting(), "Scripting is not supported for postprocess."
            if results is None:
                return None, inst_feat, contrast_feat, prototypes, keep_indices, sp_att, ch_att
            else:    
                return GeneralizedRCNN_DA._postprocess(results, batched_inputs, images.image_sizes), \
                    inst_feat, contrast_feat, prototypes, keep_indices, sp_att, ch_att

        return results
    
    def get_feat_for_visualize(
        self,
        batched_inputs: List[Dict[str, torch.Tensor]],
    ):
        assert not self.training

        images = self.preprocess_image(batched_inputs)
        # 
        features = self.backbone(images.tensor)

        # get attention features
        sp_att, ch_att = self.get_da_feat(features, level='image', alpha = self.alpha1, get_att = True)
        
        # get instance features from each GT proposals
        assert "proposals" in batched_inputs[0]
        proposals = [x["proposals"].to(self.device) for x in batched_inputs]
        # 
        inst_feat = self.roi_heads._forward_box(features, proposals, visualize=True)
        
        return features, inst_feat, sp_att, ch_att


    def preprocess_image(self, batched_inputs: List[Dict[str, torch.Tensor]]):
        """
        Normalize, pad and batch the input images.
        """
        images = [self._move_to_current_device(x["image"]) for x in batched_inputs]
        images = [(x - self.pixel_mean) / self.pixel_std for x in images]
        images = ImageList.from_tensors(
            images,
            self.backbone.size_divisibility,
            padding_constraints=self.backbone.padding_constraints,
        )
        return images

    @staticmethod
    def _postprocess(instances, batched_inputs: List[Dict[str, torch.Tensor]], image_sizes):
        """
        Rescale the output instances to the target size.
        """
        # note: private function; subject to changes
        processed_results = []
        for results_per_image, input_per_image, image_size in zip(
            instances, batched_inputs, image_sizes
        ):
            height = input_per_image.get("height", image_size[0])
            width = input_per_image.get("width", image_size[1])
            r = detector_postprocess(results_per_image, height, width)
            processed_results.append({"instances": r})
        return processed_results
    
    
    def get_da_feat(self, features, level, alpha, get_att = False):
        if level == 'image':
            if self.adapt_level in ['sup', 'inst']: return None, None
            disc_list = self.img_disc_list
            da_apply_layer = self.da_apply_layer
            
        elif level == 'inst':
            if self.adapt_level in ['sup', 'image']: return None, None
            disc_list = self.inst_disc_list
            da_apply_layer = self.da_apply_layer
        
        disc_feat_list = []
        cons_feat_list = []
        sp_att_list = []
        ch_att_list = []
        for layer_name, layer_feat in features.items():
            if layer_name not in da_apply_layer or layer_feat.nelement() == 0:
                continue
            
            # 0. apply Residual CBAM if needed
            if self.cbam:
                cbam_feat, ch_att, sp_att = self.cbam_list[layer_name](layer_feat)
                sp_att_list.append(sp_att)
                ch_att_list.append(ch_att)
                layer_feat = layer_feat + cbam_feat
                
            # 1. Discriminator feature
            rev_disc_feat = ReverseLayerF.apply(layer_feat, alpha)
            disc_feat = disc_list[layer_name](rev_disc_feat)
            disc_feat_list.append(disc_feat)
            
            # 2. Consistency feature
            rev_cons_feat = ScaleLayerF.apply(layer_feat, self.cons_weight * alpha)
            cons_feat = disc_list[layer_name](rev_cons_feat)
            
            cons_feat_list.append(cons_feat)
        
        if get_att: return sp_att_list, ch_att_list
        return disc_feat_list, cons_feat_list