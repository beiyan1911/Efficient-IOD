# Copyright (c) OpenMMLab. All rights reserved.
from typing import List, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F
from mmcv.ops import batched_nms
from torch import Tensor

from mmdet.registry import MODELS
from mmdet.structures import SampleList
from mmdet.structures.bbox import bbox_overlaps
from mmdet.utils import (ConfigType, InstanceList, OptInstanceList, reduce_mean)
from mmdet.utils import (MultiConfig, OptConfigType)
from .gfl_head_fd_tal import GFLHeadFDTAL
from ..utils import (multi_apply, unpack_gt_instances)


class Integral(nn.Module):
    """A fixed layer for calculating integral result from distribution.

    This layer calculates the target location by :math: ``sum{P(y_i) * y_i}``,
    P(y_i) denotes the softmax vector that represents the discrete distribution
    y_i denotes the discrete set, usually {0, 1, 2, ..., reg_max}

    Args:
        reg_max (int): The maximal value of the discrete set. Defaults to 16.
            You may want to reset it according to your new dataset or related
            settings.
    """

    def __init__(self, reg_max: int = 16) -> None:
        super().__init__()
        self.reg_max = reg_max
        self.register_buffer('project',
                             torch.linspace(0, self.reg_max, self.reg_max + 1))

    def forward(self, x: Tensor) -> Tensor:
        """Forward feature from the regression head to get integral result of
        bounding box location.

        Args:
            x (Tensor): Features of the regression head, shape (N, 4*(n+1)),
                n is self.reg_max.

        Returns:
            x (Tensor): Integral result of box locations, i.e., distance
                offsets from the box center in four directions, shape (N, 4).
        """
        x = F.softmax(x.reshape(-1, self.reg_max + 1), dim=1)
        x = F.linear(x, self.project.type_as(x)).reshape(-1, 4)
        return x


@MODELS.register_module()
class GFLHeadFDIncrementADGTAL(GFLHeadFDTAL):
    """Generalized Focal Loss: Learning Qualified and Distributed Bounding
    Boxes for Dense Object Detection.

    GFL head structure is similar with ATSS, however GFL uses
    1) joint representation for classification and localization quality, and
    2) flexible General distribution for bounding box locations,
    which are supervised by
    Quality Focal Loss (QFL) and Distribution Focal Loss (DFL), respectively

    https://arxiv.org/abs/2006.04388

    Args:
        num_classes (int): Number of categories excluding the background
            category.
        in_channels (int): Number of channels in the input feature map.
        stacked_convs (int): Number of conv layers in cls and reg tower.
            Defaults to 4.
        conv_cfg (:obj:`ConfigDict` or dict, optional): dictionary to construct
            and config conv layer. Defaults to None.
        norm_cfg (:obj:`ConfigDict` or dict): dictionary to construct and
            config norm layer. Default: dict(type='GN', num_groups=32,
            requires_grad=True).
        loss_qfl (:obj:`ConfigDict` or dict): Config of Quality Focal Loss
            (QFL).
        bbox_coder (:obj:`ConfigDict` or dict): Config of bbox coder. Defaults
             to 'DistancePointBBoxCoder'.
        reg_max (int): Max value of integral set :math: ``{0, ..., reg_max}``
            in QFL setting. Defaults to 16.
        init_cfg (:obj:`ConfigDict` or dict or list[dict] or
            list[:obj:`ConfigDict`]): Initialization config dict.
    Example:
        >>> self = GFLHead(11, 7)
        >>> feats = [torch.rand(1, 7, s, s) for s in [4, 8, 16, 32, 64]]
        >>> cls_quality_score, bbox_pred = self.forward(feats)
        >>> assert len(cls_quality_score) == len(self.scales)
    """

    def __init__(self,
                 num_tasks,
                 incre_cls_nums,
                 num_classes: int,
                 in_channels: int,
                 stacked_convs: int = 4,
                 conv_cfg: OptConfigType = None,
                 norm_cfg: ConfigType = dict(
                     type='GN', num_groups=32, requires_grad=True),
                 loss_dfl: ConfigType = dict(
                     type='DistributionFocalLoss', loss_weight=0.25),
                 loss_ld: ConfigType = dict(
                     type='KnowledgeDistillationKLDivLoss', loss_weight=0.25, T=10),
                 bbox_coder: ConfigType = dict(type='DistancePointBBoxCoder'),
                 reg_max: int = 16,
                 init_cfg: MultiConfig = dict(
                     type='Normal',
                     layer='Conv2d',
                     std=0.01,
                     override=dict(
                         type='Normal',
                         name='gfl_cls_list',
                         std=0.01,
                         bias_prob=0.01)),
                 initial_loss_cls: ConfigType = dict(
                     type='FocalLoss',
                     use_sigmoid=True,
                     activated=True,
                     gamma=2.0,
                     alpha=0.25,
                     loss_weight=1.0),
                 **kwargs) -> None:
        self.num_tasks = num_tasks
        self.incre_cls_nums = incre_cls_nums
        self.stacked_convs = stacked_convs
        self.conv_cfg = conv_cfg
        self.norm_cfg = norm_cfg
        self.reg_max = reg_max
        super().__init__(
            num_tasks=num_tasks,
            incre_cls_nums=incre_cls_nums,
            num_classes=num_classes,
            in_channels=in_channels,
            bbox_coder=bbox_coder,
            init_cfg=init_cfg,
            **kwargs)

        self.loss_dist_cls_weight = 5.0  # L1
        self.loss_dist_cls = nn.L1Loss(reduction='none')  # L1
        self.loss_dist_bbox = MODELS.build(dict(type='GIoULoss', loss_weight=1.0))
        self.loss_dist_bbox_kl = MODELS.build(dict(type='KnowledgeDistillationKLDivLoss', loss_weight=1.0, T=10))

    def distill_loss_by_image_single(self,
                                     anchors,
                                     new_cls_preds,
                                     new_bbox_preds,
                                     new_bbox_decoded,
                                     ori_cls_inds,
                                     ori_box_inds,
                                     tea_cls_preds,
                                     tea_bbox_preds,
                                     positive_idx,
                                     flatten_strides,
                                     dist_loss_weight,
                                     ori_num_classes: int, avg_factor: int) -> dict:
        """Calculate the loss of a single scale level based on the features
        extracted by the detection head.

        Args:
            anchors (Tensor): Box reference for each scale level with shape
                (N, num_total_anchors, 4).
            cls_score (Tensor): Cls and quality joint scores for each scale
                level has shape (N, num_classes, H, W).
            bbox_pred (Tensor): Box distribution logits for each scale
                level with shape (N, 4*(n+1), H, W), n is max value of integral
                set.
            labels (Tensor): Labels of each anchors with shape
                (N, num_total_anchors).
            label_weights (Tensor): Label weights of each anchor with shape
                (N, num_total_anchors)
            bbox_targets (Tensor): BBox regression targets of each anchor
                weight shape (N, num_total_anchors, 4).
            stride (Tuple[int]): Stride in this scale level.
            avg_factor (int): Average factor that is used to average
                the loss. When using sampling method, avg_factor is usually
                the sum of positive and negative priors. When using
                `PseudoSampler`, `avg_factor` is usually equal to the number
                of positive priors.

        Returns:
            dict[str, Tensor]: A dictionary of loss components.
        """

        # ----------------- distillation for cls branch -----------------
        tea_cls_scores = tea_cls_preds.sigmoid()
        new_cls_scores = new_cls_preds.sigmoid()
        tea_cls_conf, tea_cls_label = tea_cls_scores.max(-1)
        # new_cls_conf, _ = new_cls_scores.max(-1)

        focal_weight_cls = tea_cls_conf
        focal_weight_reg = tea_cls_conf

        loss_dist_cls = (self.loss_dist_cls(new_cls_scores, tea_cls_scores).sum(
            -1) * focal_weight_cls).sum() * self.loss_dist_cls_weight

        # ----------------- distillation for reg branch -----------------

        ori_box_inds = ori_cls_inds
        ori_box_inds = ori_box_inds[torch.isin(ori_box_inds, positive_idx, invert=True)]

        topk_bbox_targets = tea_bbox_preds[ori_box_inds]
        topk_bbox_pred = new_bbox_preds[ori_box_inds]

        topk_anchors = anchors[ori_box_inds]
        topk_strides = flatten_strides[ori_box_inds]
        # decode new pred boxes
        topk_anchor_centers = self.anchor_center(topk_anchors)

        topk_decode_bbox_pred = new_bbox_decoded[ori_box_inds]
        # decode tea pred boxes
        topk_bbox_target_corners = self.integral(topk_bbox_targets) * topk_strides[:, 0:1]
        topk_decode_bbox_target = self.bbox_coder.decode(
            topk_anchor_centers, topk_bbox_target_corners)

        topk_bbox_weight, topk_labels = focal_weight_reg[ori_box_inds], tea_cls_label[ori_box_inds]

        # nms
        nms_cfg = dict(iou_threshold=0.05)  # 0.005
        _, keep = batched_nms(topk_decode_bbox_target, topk_bbox_weight, topk_labels, nms_cfg)

        # nms filter
        nms_target_bbox = topk_decode_bbox_target.gather(
            0, keep.unsqueeze(-1).expand(-1, topk_decode_bbox_target.size(-1)))
        nms_predict_bbox = topk_decode_bbox_pred.gather(
            0, keep.unsqueeze(-1).expand(-1, topk_decode_bbox_pred.size(-1)))

        nms_weight = topk_bbox_weight[keep]

        loss_dist_bbox_kl = self.loss_dist_bbox_kl(
            topk_bbox_pred[keep].reshape(-1, self.reg_max + 1),
            topk_bbox_targets[keep].reshape(-1, self.reg_max + 1),
            weight=nms_weight[:, None].expand(-1, 4).reshape(-1),
            avg_factor=4.0)

        loss_dist_bbox = self.loss_dist_bbox(
            nms_predict_bbox,
            nms_target_bbox,
            weight=nms_weight,
            avg_factor=1.0)

        return loss_dist_cls, loss_dist_bbox, loss_dist_bbox_kl, focal_weight_cls.sum(), nms_weight.sum()

    def loss_by_feat_single(self, anchors: Tensor, cls_pred: Tensor,
                            bbox_pred: Tensor, labels: Tensor,
                            label_weights: Tensor, bbox_targets: Tensor, alignment_metrics: Tensor,
                            stride: Tuple[int], ori_num_classes: int) -> dict:
        """Calculate the loss of a single scale level based on the features
        extracted by the detection head.

        Args:
            anchors (Tensor): Box reference for each scale level with shape
                (N, num_total_anchors, 4).
            cls_pred (Tensor): Cls and quality joint scores for each scale
                level has shape (N, num_classes, H, W).
            bbox_pred (Tensor): Box distribution logits for each scale
                level with shape (N, 4*(n+1), H, W), n is max value of integral
                set.
            labels (Tensor): Labels of each anchors with shape
                (N, num_total_anchors).
            label_weights (Tensor): Label weights of each anchor with shape
                (N, num_total_anchors)
            bbox_targets (Tensor): BBox regression targets of each anchor
                weight shape (N, num_total_anchors, 4).
            stride (Tuple[int]): Stride in this scale level.
            avg_factor (int): Average factor that is used to average
                the loss. When using sampling method, avg_factor is usually
                the sum of positive and negative priors. When using
                `PseudoSampler`, `avg_factor` is usually equal to the number
                of positive priors.

        Returns:
            dict[str, Tensor]: A dictionary of loss components.
        """
        assert stride[0] == stride[1], 'h stride is not equal to w stride!'
        anchors = anchors.reshape(-1, 4)
        # cls_score = cls_score.permute(0, 2, 3,
        #                               1).reshape(-1, self.cls_out_channels)
        cls_score = cls_pred[:, ori_num_classes:].permute(0, 2, 3,
                                                          1).reshape(-1,
                                                                     self.cls_out_channels - ori_num_classes).sigmoid()

        bbox_pred = bbox_pred.permute(0, 2, 3,
                                      1).reshape(-1, 4 * (self.reg_max + 1))
        bbox_targets = bbox_targets.reshape(-1, 4)
        labels = labels.reshape(-1)
        alignment_metrics = alignment_metrics.reshape(-1)
        label_weights = label_weights.reshape(-1)

        # FG cat_id: [0, num_classes -1], BG cat_id: num_classes
        bg_class_ind = self.num_classes - ori_num_classes  # only optimize the novel classes
        labels[labels == self.num_classes] = bg_class_ind  # only optimize the novel classes

        pos_inds = ((labels >= 0)
                    & (labels < bg_class_ind)).nonzero().squeeze(1)
        score = label_weights.new_zeros(labels.shape)

        if len(pos_inds) > 0:
            pos_bbox_targets = bbox_targets[pos_inds]
            pos_bbox_pred = bbox_pred[pos_inds]
            pos_anchors = anchors[pos_inds]
            pos_anchor_centers = self.anchor_center(pos_anchors) / stride[0]

            # weight_targets = cls_pred.detach().sigmoid()
            # weight_targets = weight_targets.max(dim=1)[0][pos_inds]

            weight_targets = self.centerness_target(
                pos_anchors, pos_bbox_targets
            ) if self.epoch < self.initial_epoch else alignment_metrics[
                pos_inds]

            pos_bbox_pred_corners = self.integral(pos_bbox_pred)
            pos_decode_bbox_pred = self.bbox_coder.decode(
                pos_anchor_centers, pos_bbox_pred_corners)
            pos_decode_bbox_targets = pos_bbox_targets / stride[0]
            score[pos_inds] = bbox_overlaps(
                pos_decode_bbox_pred.detach(),
                pos_decode_bbox_targets,
                is_aligned=True)
            pred_corners = pos_bbox_pred.reshape(-1, self.reg_max + 1)
            target_corners = self.bbox_coder.encode(pos_anchor_centers,
                                                    pos_decode_bbox_targets,
                                                    self.reg_max).reshape(-1)

            # regression loss
            loss_bbox = self.loss_bbox(
                pos_decode_bbox_pred,
                pos_decode_bbox_targets,
                weight=weight_targets,
                avg_factor=1.0)

            # dfl loss
            loss_dfl = self.loss_dfl(
                pred_corners,
                target_corners,
                weight=weight_targets[:, None].expand(-1, 4).reshape(-1),
                avg_factor=4.0)
        else:
            loss_bbox = bbox_pred.sum() * 0
            loss_dfl = bbox_pred.sum() * 0
            weight_targets = bbox_pred.new_tensor(0)

        # cls (qfl) loss
        cls_targets = labels if self.epoch < self.initial_epoch else (
            labels, alignment_metrics)
        cls_loss_func = self.initial_loss_cls \
            if self.epoch < self.initial_epoch else self.loss_cls

        loss_cls = cls_loss_func(
            cls_score, cls_targets, label_weights, avg_factor=1.0)

        return loss_cls, loss_bbox, loss_dfl, alignment_metrics.sum(), weight_targets.sum()

    def loss_by_feat(self,
                     ori_outs: Tuple[Tensor],
                     new_outs: Tuple[Tensor],
                     ori_topk_cls_inds,  # for distillation
                     ori_topk_bbox_inds,  # for distillation
                     ori_num_classes,
                     dist_loss_weight,
                     model,
                     batch_gt_instances: InstanceList,
                     batch_img_metas: List[dict],
                     batch_gt_instances_ignore: OptInstanceList = None) -> dict:
        """Calculate the loss based on the features extracted by the detection
        head.

        Args:
            cls_preds (list[Tensor]): Cls and quality scores for each scale
                level has shape (N, num_classes, H, W).
            bbox_preds (list[Tensor]): Box distribution logits for each scale
                level with shape (N, 4*(n+1), H, W), n is max value of integral
                set.
            batch_gt_instances (list[:obj:`InstanceData`]): Batch of
                gt_instance.  It usually includes ``bboxes`` and ``labels``
                attributes.
            batch_img_metas (list[dict]): Meta information of each image, e.g.,
                image size, scaling factor, etc.
            batch_gt_instances_ignore (list[:obj:`InstanceData`], Optional):
                Batch of gt_instances_ignore. It includes ``bboxes`` attribute
                data that is ignored during training and testing.
                Defaults to None.

        Returns:
            dict[str, Tensor]: A dictionary of loss components.
        """
        # ****************************** ori loss **********************************
        cls_preds, bbox_preds = new_outs
        # cls_scores = [cls_pred.sigmoid() for cls_pred in cls_preds]

        num_imgs = len(batch_img_metas)
        featmap_sizes = [featmap.size()[-2:] for featmap in cls_preds]
        assert len(featmap_sizes) == self.prior_generator.num_levels

        device = cls_preds[0].device
        anchor_list, valid_flag_list = self.get_anchors(
            featmap_sizes, batch_img_metas, device=device)

        flatten_cls_preds = torch.cat([
            cls_score.permute(0, 2, 3, 1).reshape(num_imgs, -1, self.cls_out_channels)
            for cls_score in cls_preds
        ], 1)

        flatten_bbox_preds = torch.cat([
            bbox_pred.permute(0, 2, 3, 1).reshape(num_imgs, -1, 4 * (self.reg_max + 1))
            for bbox_pred in bbox_preds
        ], 1)

        flatten_strides = []
        for _idx, _per in enumerate(featmap_sizes):
            flatten_strides.extend(_per.numel() * [self.prior_generator.strides[_idx]])
        flatten_strides = flatten_bbox_preds.new(flatten_strides)
        flatten_strides = torch.stack([flatten_strides for _ in range(num_imgs)])

        flatten_anchors = torch.stack([torch.cat(per_anchors, dim=0) for per_anchors in anchor_list])

        flatten_anchors_centers = self.anchor_center(flatten_anchors)
        real_flatten_bbox_preds = self.bbox_coder.decode(flatten_anchors_centers,
                                                         self.integral(flatten_bbox_preds).
                                                         reshape(num_imgs, -1, 4) *
                                                         flatten_strides[:, :, 0:1])

        cls_reg_targets = self.get_targets(
            flatten_cls_preds[:, :, ori_num_classes:].sigmoid(),
            real_flatten_bbox_preds,
            anchor_list,
            valid_flag_list,
            batch_gt_instances,
            batch_img_metas,
            batch_gt_instances_ignore=batch_gt_instances_ignore)

        (anchor_list, labels_list, label_weights_list, bbox_targets_list,
         alignment_metrics_list) = cls_reg_targets

        positive_idx_list = [
            ((single_labels >= 0) & (single_labels < self.num_classes)).nonzero(as_tuple=False).squeeze(1) for
            single_labels in torch.cat(labels_list, dim=1)]

        losses_cls, losses_bbox, losses_dfl, \
        cls_avg_factor, avg_factor = multi_apply(
            self.loss_by_feat_single,
            anchor_list,
            cls_preds,
            bbox_preds,
            labels_list,
            label_weights_list,
            bbox_targets_list,
            alignment_metrics_list,
            self.prior_generator.strides,
            ori_num_classes=ori_num_classes)
        # avg_factor=avg_factor)

        avg_factor = sum(avg_factor)
        avg_factor = reduce_mean(avg_factor).clamp_(min=1).item()

        cls_avg_factor = sum(cls_avg_factor)
        cls_avg_factor = reduce_mean(cls_avg_factor).clamp_(min=1).item()
        losses_bbox = list(map(lambda x: x / avg_factor, losses_bbox))
        losses_cls = list(map(lambda x: x / cls_avg_factor, losses_cls))
        losses_dfl = list(map(lambda x: x / avg_factor, losses_dfl))

        # ****************************** distill loss **********************************
        anchor_list = torch.cat(anchor_list, dim=1)
        ori_cls_preds, ori_bbox_preds = ori_outs

        ori_cls_preds_list = [
            ori_cls_pred[:, :ori_num_classes, :, :].permute(0, 2, 3, 1).reshape(
                num_imgs, -1, ori_num_classes)
            for ori_cls_pred in ori_cls_preds]
        ori_cls_preds_list = torch.cat(ori_cls_preds_list, dim=1)

        ori_bbox_preds_list = [
            ori_bbox_pred.permute(0, 2, 3, 1).reshape(num_imgs, -1, 4 * (self.reg_max + 1))
            for ori_bbox_pred in ori_bbox_preds]
        ori_bbox_preds_list = torch.cat(ori_bbox_preds_list, dim=1)

        # Select the results corresponding to the new model and the old model
        new_cls_scores_list = flatten_cls_preds[:, :, :ori_num_classes]

        loss_dist_cls, loss_dist_bbox, loss_dist_bbox_kl, cls_dist_avg_factor, box_dist_avg_factor = \
            multi_apply(
                self.distill_loss_by_image_single,
                anchor_list,
                new_cls_scores_list,  # new model: cls new part
                flatten_bbox_preds,  # new model: dist reg
                real_flatten_bbox_preds,  # new model: bbox
                ori_topk_cls_inds,
                ori_topk_bbox_inds,
                ori_cls_preds_list,  # ori model: cls ori part
                ori_bbox_preds_list,  # ori model: dist reg
                positive_idx_list,
                flatten_strides,
                dist_loss_weight=dist_loss_weight,
                ori_num_classes=ori_num_classes,
                avg_factor=avg_factor)

        dist_cls_avg_factors = reduce_mean(sum(cls_dist_avg_factor)).clamp_(min=1).item()
        loss_dist_cls = list(map(lambda x: x / dist_cls_avg_factors, loss_dist_cls))

        box_dist_avg_factor = reduce_mean(sum(box_dist_avg_factor)).clamp_(min=1).item()
        loss_dist_bbox = list(map(lambda x: x / box_dist_avg_factor, loss_dist_bbox))
        # loss_dist_bbox_kl = list(map(lambda x: x / box_dist_avg_factor, loss_dist_bbox_kl))

        return dict(
            loss_cls=losses_cls,
            loss_bbox=losses_bbox,
            loss_dfl=losses_dfl,
            loss_dist_cls=loss_dist_cls,
            loss_dist_bbox=loss_dist_bbox,
            loss_dist_bbox_kl=loss_dist_bbox_kl)

    def predict_for_visualization(self,
                                  x: Tuple[Tensor],
                                  batch_data_samples: SampleList,
                                  rescale: bool = False):
        """Perform forward propagation of the detection head and predict
        detection results on the features of the upstream network.

        Args:
            x (tuple[Tensor]): Multi-level features from the
                upstream network, each is a 4D-tensor.
            batch_data_samples (List[:obj:`DetDataSample`]): The Data
                Samples. It usually includes information such as
                `gt_instance`, `gt_panoptic_seg` and `gt_sem_seg`.
            rescale (bool, optional): Whether to rescale the results.
                Defaults to False.

        Returns:
            list[obj:`InstanceData`]: Detection results of each image
            after the post process.
        """
        batch_img_metas = [
            data_samples.metainfo for data_samples in batch_data_samples
        ]

        outs = self(x)

        predictions = self.predict_by_feat(
            *outs, batch_img_metas=batch_img_metas, rescale=rescale)
        return predictions, outs


    # def loss(self, ori_out: Tuple[Tensor], new_out: Tuple[Tensor],batch_data_samples: SampleList) -> dict:
    def loss(self, ori_outs: Tuple[Tensor], new_outs: Tuple[Tensor], batch_data_samples: SampleList,
             topk_cls_inds, topk_bbox_inds,
             ori_num_classes, dist_loss_weight, model) -> dict:
        """Perform forward propagation and loss calculation of the detection
        head on the features of the upstream network.

        Args:
            x (tuple[Tensor]): Features from the upstream network, each is
                a 4D-tensor.
            batch_data_samples (List[:obj:`DetDataSample`]): The Data
                Samples. It usually includes information such as
                `gt_instance`, `gt_panoptic_seg` and `gt_sem_seg`.

        Returns:
            dict: A dictionary of loss components.
        """
        # outs = self(x)

        outputs = unpack_gt_instances(batch_data_samples)
        (batch_gt_instances, batch_gt_instances_ignore,
         batch_img_metas) = outputs

        loss_inputs = (ori_outs, new_outs, topk_cls_inds, topk_bbox_inds,
                       ori_num_classes, dist_loss_weight, model) + (
                          batch_gt_instances, batch_img_metas,
                          batch_gt_instances_ignore)
        losses = self.loss_by_feat(*loss_inputs)
        return losses



    def loss_for_visualization(self, new_outs: Tuple[Tensor], batch_data_samples: SampleList,
             ori_num_classes, dist_loss_weight, model) -> dict:
        """Perform forward propagation and loss calculation of the detection
        head on the features of the upstream network.

        Args:
            x (tuple[Tensor]): Features from the upstream network, each is
                a 4D-tensor.
            batch_data_samples (List[:obj:`DetDataSample`]): The Data
                Samples. It usually includes information such as
                `gt_instance`, `gt_panoptic_seg` and `gt_sem_seg`.

        Returns:
            dict: A dictionary of loss components.
        """
        # outs = self(x)
        outputs = unpack_gt_instances(batch_data_samples)
        (batch_gt_instances, batch_gt_instances_ignore,
         batch_img_metas) = outputs

        loss_inputs = ( new_outs,
                       ori_num_classes, dist_loss_weight, model) + (
                          batch_gt_instances, batch_img_metas,
                          batch_gt_instances_ignore)
        overlaps_list = self.loss_by_feat_for_visualization(*loss_inputs)
        return overlaps_list

    def loss_by_feat_for_visualization(self,
                     new_outs: Tuple[Tensor],
                     ori_num_classes,
                     dist_loss_weight,
                     model,
                     batch_gt_instances: InstanceList,
                     batch_img_metas: List[dict],
                     batch_gt_instances_ignore: OptInstanceList = None) -> dict:
        """Calculate the loss based on the features extracted by the detection
        head.

        Args:
            cls_preds (list[Tensor]): Cls and quality scores for each scale
                level has shape (N, num_classes, H, W).
            bbox_preds (list[Tensor]): Box distribution logits for each scale
                level with shape (N, 4*(n+1), H, W), n is max value of integral
                set.
            batch_gt_instances (list[:obj:`InstanceData`]): Batch of
                gt_instance.  It usually includes ``bboxes`` and ``labels``
                attributes.
            batch_img_metas (list[dict]): Meta information of each image, e.g.,
                image size, scaling factor, etc.
            batch_gt_instances_ignore (list[:obj:`InstanceData`], Optional):
                Batch of gt_instances_ignore. It includes ``bboxes`` attribute
                data that is ignored during training and testing.
                Defaults to None.

        Returns:
            dict[str, Tensor]: A dictionary of loss components.
        """
        # ****************************** ori loss **********************************
        cls_preds, bbox_preds = new_outs
        # cls_scores = [cls_pred.sigmoid() for cls_pred in cls_preds]

        num_imgs = len(batch_img_metas)
        featmap_sizes = [featmap.size()[-2:] for featmap in cls_preds]
        assert len(featmap_sizes) == self.prior_generator.num_levels

        device = cls_preds[0].device
        anchor_list, valid_flag_list = self.get_anchors(
            featmap_sizes, batch_img_metas, device=device)

        flatten_cls_preds = torch.cat([
            cls_score.permute(0, 2, 3, 1).reshape(num_imgs, -1, self.cls_out_channels)
            for cls_score in cls_preds
        ], 1)

        flatten_bbox_preds = torch.cat([
            bbox_pred.permute(0, 2, 3, 1).reshape(num_imgs, -1, 4 * (self.reg_max + 1))
            for bbox_pred in bbox_preds
        ], 1)

        flatten_strides = []
        for _idx, _per in enumerate(featmap_sizes):
            flatten_strides.extend(_per.numel() * [self.prior_generator.strides[_idx]])
        flatten_strides = flatten_bbox_preds.new(flatten_strides)
        flatten_strides = torch.stack([flatten_strides for _ in range(num_imgs)])

        flatten_anchors = torch.stack([torch.cat(per_anchors, dim=0) for per_anchors in anchor_list])

        flatten_anchors_centers = self.anchor_center(flatten_anchors)
        real_flatten_bbox_preds = self.bbox_coder.decode(flatten_anchors_centers,
                                                         self.integral(flatten_bbox_preds).
                                                         reshape(num_imgs, -1, 4) *
                                                         flatten_strides[:, :, 0:1])

        flatten_overlaps = self.get_targets_for_visualization(
            flatten_cls_preds.sigmoid(),
            real_flatten_bbox_preds,
            anchor_list,
            valid_flag_list,
            batch_gt_instances,
            batch_img_metas,
            batch_gt_instances_ignore=batch_gt_instances_ignore)

        flatten_overlaps = flatten_overlaps[0].max(1)[0]
        count_list = [bbox_pred.shape[2:].numel() for bbox_pred in bbox_preds]
        shape_list = [bbox_pred.shape[2:] for bbox_pred in bbox_preds]
        _start = 0
        overlaps_list = []
        for _count, _shape in zip(count_list, shape_list):
            slice_overlaps = flatten_overlaps[_start:_start + _count]
            overlaps_list.append(slice_overlaps.reshape(_shape))
            _start = _start + _count
        return overlaps_list
