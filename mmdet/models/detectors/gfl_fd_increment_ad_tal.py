# Copyright (c) OpenMMLab. All rights reserved.
import os
from collections import OrderedDict
from typing import List, Union

import torch
from mmengine import Config
from mmengine.registry import (MODELS)
from mmengine.runner.checkpoint import load_checkpoint, load_state_dict

from torch import Tensor
from ..utils import multi_apply
from mmdet.registry import MODELS
from mmdet.structures import SampleList
from mmdet.utils import ConfigType, OptConfigType, OptMultiConfig

from .gfl import GFL

from typing import Dict, Union

# ******** begin for visualization
from typing import List, Tuple
from mmdet.structures import DetDataSample, OptSampleList

ForwardResults = Union[Dict[str, torch.Tensor], List[DetDataSample],
Tuple[torch.Tensor], torch.Tensor]


# ******** end for visualization


@MODELS.register_module()
class GFLFDIncrementADTAL(GFL):
    """Implementation of `GFL <https://arxiv.org/abs/2006.04388>`_

    Args:
        backbone (:obj:`ConfigDict` or dict): The backbone module.
        neck (:obj:`ConfigDict` or dict): The neck module.
        bbox_head (:obj:`ConfigDict` or dict): The bbox head module.
        train_cfg (:obj:`ConfigDict` or dict, optional): The training config
            of GFL. Defaults to None.
        test_cfg (:obj:`ConfigDict` or dict, optional): The testing config
            of GFL. Defaults to None.
        data_preprocessor (:obj:`ConfigDict` or dict, optional): Config of
            :class:`DetDataPreprocessor` to process the input data.
            Defaults to None.
        init_cfg (:obj:`ConfigDict` or dict, optional): the config to control
            the initialization. Defaults to None.
    """

    def __init__(self,
                 ori_setting: ConfigType,
                 backbone: ConfigType,
                 neck: ConfigType,
                 bbox_head: ConfigType,
                 train_cfg: OptConfigType = None,
                 test_cfg: OptConfigType = None,
                 data_preprocessor: OptConfigType = None,
                 init_cfg: OptMultiConfig = None,
                 latest_model_flag=True,
                 top_k=100,
                 dist_loss_weight=1,
                 ) -> None:
        super().__init__(
            backbone=backbone,
            neck=neck,
            bbox_head=bbox_head,
            train_cfg=train_cfg,
            test_cfg=test_cfg,
            data_preprocessor=data_preprocessor,
            init_cfg=init_cfg)

        self.top_k = top_k
        self.dist_loss_weight = dist_loss_weight
        if latest_model_flag:
            self.load_base_detector(ori_setting)
            self._is_init = True

    def _load_checkpoint_for_new_model(self, checkpoint_file, map_location=None, strict=False, logger=None):
        self.bbox_head.init_weights()
        # load ckpt
        checkpoint = torch.load(checkpoint_file, map_location=map_location)
        # get state_dict from checkpoint
        if isinstance(checkpoint, OrderedDict):
            state_dict = checkpoint
        elif isinstance(checkpoint, dict) and 'state_dict' in checkpoint:
            state_dict = checkpoint['state_dict']
        else:
            raise RuntimeError(
                'No state_dict found in checkpoint file {}'.format(checkpoint_file))
        # strip prefix of state_dict
        if list(state_dict.keys())[0].startswith('module.'):
            state_dict = {k[7:]: v for k,
            v in checkpoint['state_dict'].items()}
        # modify cls head size of state_dict
        # added_branch_weight = self.bbox_head.gfl_cls.weight[self.ori_num_classes:, ...]
        # added_branch_bias = self.bbox_head.gfl_cls.bias[self.ori_num_classes:, ...]
        # state_dict['bbox_head.gfl_cls.weight'] = torch.cat(
        #     (state_dict['bbox_head.gfl_cls.weight'], added_branch_weight), dim=0)
        # state_dict['bbox_head.gfl_cls.bias'] = torch.cat(
        #     (state_dict['bbox_head.gfl_cls.bias'], added_branch_bias), dim=0)
        # load state_dict
        if hasattr(self, 'module'):
            load_state_dict(self.module, state_dict, strict, logger)
        else:
            load_state_dict(self, state_dict, strict, logger)

    def load_base_detector(self, ori_setting):
        """
                Initialize detector from config file.
        :param ori_setting:
        :return:
        """
        assert os.path.isfile(ori_setting['ori_checkpoint_file']), '{} is not a valid file'.format(
            ori_setting['ori_checkpoint_file'])
        ##### init original model & frozen it #####
        # build model
        ori_cfg = Config.fromfile(ori_setting['ori_config_file'])
        if hasattr(ori_cfg.model, 'latest_model_flag'):
            ori_cfg.model.latest_model_flag = False
        ori_model = MODELS.build(ori_cfg.model)
        # load checkpoint
        load_checkpoint(ori_model, ori_setting.ori_checkpoint_file, strict=False)
        # # set to eval mode
        ori_model.eval()
        # ori_model.forward = ori_model.forward_dummy
        # # set requires_grad of all parameters to False
        for param in ori_model.parameters():
            param.requires_grad = False

        # ##### init original branchs of new model #####
        self.ori_num_classes = ori_setting.ori_num_classes
        self._load_checkpoint_for_new_model(ori_setting.ori_checkpoint_file)
        print('======> load base checkpoint for new model from {}'.format(ori_setting.ori_checkpoint_file))
        self.ori_model = ori_model

    def forward_ori_model(self, img):
        """Inference image(s) with the detector.

        Args:
            model (nn.Module): The loaded detector.
            img (Tensor): Input to the model.

        Returns:
            outs (Tuple(List[Tensor])): Three model outputs.
                # cls_scores (List[Tensor]): Classification scores for each FPN level.
                # bbox_preds (List[Tensor]): BBox predictions for each FPN level.
                # centernesses (List[Tensor]): Centernesses predictions for each FPN level.
        """
        # forward the model without gradients
        with torch.no_grad():
            outs = self.ori_model(img)

        return outs

    # ********************************** begin for visualization ***********************************

    def test_step_for_visualization(self, data: Union[dict, tuple, list]) -> list:
        """``BaseModel`` implements ``test_step`` the same as ``val_step``.

        Args:
            data (dict or tuple or list): Data sampled from dataset.

        Returns:
            list: The predictions of given data.
        """
        data = self.data_preprocessor(data, False)
        overlaps_list, batch_data_samples, output = self._run_forward_for_visualization(data,
                                                                                        mode='predict')  # type: ignore
        return data, overlaps_list, batch_data_samples, output

    def _run_forward_for_visualization(self, data: Union[dict, tuple, list],
                                       mode: str):
        """Unpacks data for :meth:`forward`

        Args:
            data (dict or tuple or list): Data sampled from dataset.
            mode (str): Mode of forward.

        Returns:
            dict or list: Results of training or testing mode.
        """
        # inputs: torch.Tensor,
        # data_samples: OptSampleList = None,

        overlaps_list, batch_data_samples, output = self.forward_for_visualization(data["inputs"], data["data_samples"],
                                                                                   mode=mode)
        return overlaps_list, batch_data_samples, output

    def forward_for_visualization(self,
                                  inputs: torch.Tensor,
                                  data_samples: OptSampleList = None,
                                  mode: str = 'tensor') -> ForwardResults:
        """The unified entry for a forward process in both training and test.

        The method should accept three modes: "tensor", "predict" and "loss":

        - "tensor": Forward the whole network and return tensor or tuple of
        tensor without any post-processing, same as a common nn.Module.
        - "predict": Forward and return the predictions, which are fully
        processed to a list of :obj:`DetDataSample`.
        - "loss": Forward and return a dict of losses according to the given
        inputs and data samples.

        Note that this method doesn't handle either back propagation or
        parameter update, which are supposed to be done in :meth:`train_step`.

        Args:
            inputs (torch.Tensor): The input tensor with shape
                (N, C, ...) in general.
            data_samples (list[:obj:`DetDataSample`], optional): A batch of
                data samples that contain annotations and predictions.
                Defaults to None.
            mode (str): Return what kind of value. Defaults to 'tensor'.

        Returns:
            The return type depends on ``mode``.

            - If ``mode="tensor"``, return a tensor or a tuple of tensor.
            - If ``mode="predict"``, return a list of :obj:`DetDataSample`.
            - If ``mode="loss"``, return a dict of tensor.
        """
        overlaps_list = self.loss_for_visualization(inputs, data_samples)
        batch_data_samples, output = self.predict_for_visualization(inputs, data_samples)
        return overlaps_list, batch_data_samples, output

    # ********************************** end for visualization ***********************************

    def sel_pos_single(self, cat_cls_scores, cat_bbox_preds):

        # select topk for classifation
        cat_conf = cat_cls_scores.sigmoid()
        max_scores, _ = cat_conf.max(dim=-1)

        cls_thr = max_scores.mean() + 2 * max_scores.std()
        valid_mask = max_scores > cls_thr
        topk_cls_inds = valid_mask.nonzero(as_tuple=False).squeeze(1)

        # select topk for regression
        max_bbox, _ = cat_bbox_preds.max(dim=-1)  # 4*17 = 68 中选取最大值进行筛选
        bbox_thr = max_bbox.mean() + 2 * max_bbox.std()
        bbox_valid_mask = max_bbox > bbox_thr
        topk_bbox_inds = bbox_valid_mask.nonzero(as_tuple=False).squeeze(1)

        return topk_cls_inds, topk_bbox_inds

    def sel_pos(self, cls_scores, bbox_preds):
        """Select positive predictions based on classification scores.

        Args:
            model (nn.Module): The loaded detector.
            cls_scores (List[Tensor]): Classification scores for each FPN level.
            bbox_preds (List[Tensor]): BBox predictions for each FPN level.
            #centernesses (List[Tensor]): Centernesses predictions for each FPN level.

        Returns:
            cat_cls_scores (Tensor): FPN concatenated classification scores.
            #cat_centernesses (Tensor): FPN concatenated centernesses.
            topk_bbox_preds (Tensor): Selected top-k bbox predictions.
            topk_inds (Tensor): Selected top-k indices.
        """
        assert len(cls_scores) == len(bbox_preds)

        num_imgs = cls_scores[0].size(0)
        cat_cls_scores = [
            cls_score.permute(0, 2, 3, 1).reshape(
                num_imgs, -1, self.ori_model.bbox_head.cls_out_channels)
            for cls_score in cls_scores
        ]
        cat_bbox_preds = [
            bbox_pred.permute(0, 2, 3, 1).reshape(num_imgs, -1, 4 * (self.ori_model.bbox_head.reg_max + 1))  # ori:4
            for bbox_pred in bbox_preds
        ]
        cat_cls_scores = torch.cat(cat_cls_scores, dim=1)
        cat_bbox_preds = torch.cat(cat_bbox_preds, dim=1)

        topk_cls_inds, topk_bbox_inds = multi_apply(
            self.sel_pos_single,
            cat_cls_scores,
            cat_bbox_preds)

        return topk_cls_inds, topk_bbox_inds

    def loss(self, batch_inputs: Tensor, batch_data_samples: SampleList) -> Union[dict, list]:

        # get original model outputs
        ori_outs = self.forward_ori_model(batch_inputs)

        # select positive predictions from original model
        topk_cls_inds, topk_bbox_inds = self.sel_pos(*ori_outs)

        # get new model outputs
        x = self.extract_feat(batch_inputs)
        new_outs = self.bbox_head(x)

        # calculate losses including general losses of new model and distillation losses of original model
        loss_inputs = (ori_outs, new_outs, batch_data_samples, \
                       topk_cls_inds, topk_bbox_inds,
                       self.ori_num_classes, self.dist_loss_weight, self)

        losses = self.bbox_head.loss(*loss_inputs)
        return losses

    def loss_for_visualization(self, batch_inputs: Tensor, batch_data_samples: SampleList) -> Union[dict, list]:
        # get new model outputs
        x = self.extract_feat(batch_inputs)
        new_outs = self.bbox_head(x)

        # calculate losses including general losses of new model and distillation losses of original model
        loss_inputs = (new_outs, batch_data_samples, \
                       self.ori_num_classes, self.dist_loss_weight, self)

        overlaps_list = self.bbox_head.loss_for_visualization(*loss_inputs)
        return overlaps_list
