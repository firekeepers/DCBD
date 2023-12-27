# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved
import torch
from typing import Dict, List, Optional, Tuple, Union
# from ensemble_boxes import *
from detectron2.engine import default_argument_parser
from detectron2.structures import Boxes, ImageList, Instances, pairwise_iou
from detectron2.modeling.proposal_generator.proposal_utils import (
    add_ground_truth_to_proposals,
)
from adapteacher import add_ateacher_config
# from detectron2.modeling.box_regression import Box2BoxTransform
from detectron2.modeling.box_regression import Box2BoxTransform
from torch.nn import functional as F
from detectron2.utils.events import get_event_storage
from detectron2.modeling.roi_heads.box_head import build_box_head
from detectron2.layers import ShapeSpec
from detectron2.modeling.roi_heads import (
    ROI_HEADS_REGISTRY,
    StandardROIHeads,
)
from detectron2.layers import batched_nms
from detectron2.modeling.roi_heads.fast_rcnn import FastRCNNOutputLayers
from adapteacher.modeling.roi_heads.fast_rcnn import FastRCNNFocaltLossOutputLayers
from detectron2.config import get_cfg
from detectron2.engine import default_argument_parser, default_setup, launch
import numpy as np
from detectron2.modeling.poolers import ROIPooler
# from pesudo_box_selection import setup

# coding: utf-8
__author__ = 'ZFTurbo: https://kaggle.com/zfturbo'

import numpy as np


def bb_intersection_over_union(A, B):
    xA = max(A[0], B[0])
    yA = max(A[1], B[1])
    xB = min(A[2], B[2])
    yB = min(A[3], B[3])

    # compute the area of intersection rectangle
    interArea = max(0, xB - xA) * max(0, yB - yA)

    if interArea == 0:
        return 0.0

    # compute the area of both the prediction and ground-truth rectangles
    boxAArea = (A[2] - A[0]) * (A[3] - A[1])
    boxBArea = (B[2] - B[0]) * (B[3] - B[1])

    iou = interArea / float(boxAArea + boxBArea - interArea)
    return iou


def prefilter_boxes(boxes, scores, labels, weights, thr):
    # Create dict with boxes stored by its label
    new_boxes = dict()
    for t in range(len(boxes)):
        for j in range(len(boxes[t])):
            score = scores[t][j]
            if score < thr:
                continue
            label = int(labels[t][j])
            box_part = boxes[t][j]
            b = [int(label), float(score) * weights[t], float(box_part[0]), float(box_part[1]), float(box_part[2]),
                 float(box_part[3])]
            if label not in new_boxes:
                new_boxes[label] = []
            new_boxes[label].append(b)

    # Sort each list in dict by score and transform it to numpy array
    for k in new_boxes:
        current_boxes = np.array(new_boxes[k])
        new_boxes[k] = current_boxes[current_boxes[:, 1].argsort()[::-1]]

    return new_boxes


def get_weighted_box(boxes, conf_type='avg'):
    """
    Create weighted box for set of boxes
    :param boxes: set of boxes to fuse
    :param conf_type: type of confidence one of 'avg' or 'max'
    :return: weighted box
    """

    box = np.zeros(6, dtype=np.float32)
    conf = 0
    conf_list = []
    for b in boxes:
        box[2:] += (b[1] * b[2:])
        conf += b[1]
        conf_list.append(b[1])
    box[0] = boxes[0][0]
    if conf_type == 'avg':
        box[1] = conf / len(boxes)
    elif conf_type == 'max':
        box[1] = np.array(conf_list).max()
    box[2:] /= conf
    return box


def find_matching_box(boxes_list, new_box, match_iou):
    best_iou = match_iou
    best_index = -1
    for i in range(len(boxes_list)):
        box = boxes_list[i]
        if box[0] != new_box[0]:
            continue
        iou = bb_intersection_over_union(box[2:], new_box[2:])
        if iou > best_iou:
            best_index = i
            best_iou = iou

    return best_index, best_iou


def weighted_boxes_fusion(boxes_list, scores_list, labels_list, weights=None, iou_thr=0.55, skip_box_thr=0.8,
                          conf_type='avg', allows_overflow=False):
    '''
    :param boxes_list: list of boxes predictions from each model, each box is 4 numbers.
    It has 3 dimensions (models_number, model_preds, 4)
    Order of boxes: x1, y1, x2, y2. We expect float normalized coordinates [0; 1]
    :param scores_list: list of scores for each model
    :param labels_list: list of labels for each model
    :param weights: list of weights for each model. Default: None, which means weight == 1 for each model
    :param iou_thr: IoU value for boxes to be a match
    :param skip_box_thr: exclude boxes with score lower than this variable
    :param conf_type: how to calculate confidence in weighted boxes. 'avg': average value, 'max': maximum value
    :param allows_overflow: false if we want confidence score not exceed 1.0

    :return: boxes: boxes coordinates (Order of boxes: x1, y1, x2, y2).
    :return: scores: confidence scores
    :return: labels: boxes labels
    '''

    if weights is None:
        weights = np.ones(len(boxes_list))
    if len(weights) != len(boxes_list):
        print('Warning: incorrect number of weights {}. Must be: {}. Set weights equal to 1.'.format(len(weights),
                                                                                                     len(boxes_list)))
        weights = np.ones(len(boxes_list))
    weights = np.array(weights)

    if conf_type not in ['avg', 'max']:
        print('Unknown conf_type: {}. Must be "avg" or "max"'.format(conf_type))
        exit()

    filtered_boxes = prefilter_boxes(boxes_list, scores_list, labels_list, weights, skip_box_thr)
    if len(filtered_boxes) == 0:
        return np.zeros((0, 4)), np.zeros((0,)), np.zeros((0,))

    overall_boxes = []
    for label in filtered_boxes:
        boxes = filtered_boxes[label]
        new_boxes = []
        weighted_boxes = []

        # Clusterize boxes
        for j in range(0, len(boxes)):
            index, best_iou = find_matching_box(weighted_boxes, boxes[j], iou_thr)
            if index != -1:
                new_boxes[index].append(boxes[j])
                weighted_boxes[index] = get_weighted_box(new_boxes[index], conf_type)
            else:
                new_boxes.append([boxes[j].copy()])
                weighted_boxes.append(boxes[j].copy())

        # Rescale confidence based on number of models and boxes
        for i in range(len(new_boxes)):
            if not allows_overflow:
                weighted_boxes[i][1] = weighted_boxes[i][1] * min(weights.sum(), len(new_boxes[i])) / weights.sum()
            else:
                weighted_boxes[i][1] = weighted_boxes[i][1] * len(new_boxes[i]) / weights.sum()
        overall_boxes.append(np.array(weighted_boxes))

    overall_boxes = np.concatenate(overall_boxes, axis=0)
    overall_boxes = overall_boxes[overall_boxes[:, 1].argsort()[::-1]]
    boxes = overall_boxes[:, 2:]
    scores = overall_boxes[:, 1]
    labels = overall_boxes[:, 0]
    return boxes, scores, labels


def setup(args):
    """
    Create configs and perform basic setups.
    """
    cfg = get_cfg()
    add_ateacher_config(cfg)
    cfg.merge_from_file(args.config_file)
    cfg.merge_from_list(args.opts)
    cfg.freeze()
    default_setup(cfg, args)
    return cfg


@ROI_HEADS_REGISTRY.register()
class StandardROIHeadsPseudoLab(StandardROIHeads):
    @classmethod

    def _init_box_head(cls, cfg, input_shape):
        # fmt: off
        in_features       = cfg.MODEL.ROI_HEADS.IN_FEATURES
        pooler_resolution = cfg.MODEL.ROI_BOX_HEAD.POOLER_RESOLUTION
        pooler_scales     = tuple(1.0 / input_shape[k].stride for k in in_features)
        sampling_ratio    = cfg.MODEL.ROI_BOX_HEAD.POOLER_SAMPLING_RATIO
        pooler_type       = cfg.MODEL.ROI_BOX_HEAD.POOLER_TYPE
        # fmt: on
        # box2box_transform = Box2BoxTransform(weights=cfg.MODEL.ROI_BOX_HEAD.BBOX_REG_WEIGHTS)
        in_channels = [input_shape[f].channels for f in in_features]
        # Check all channel counts are equal
        assert len(set(in_channels)) == 1, in_channels
        in_channels = in_channels[0]

        box_pooler = ROIPooler(
            output_size=pooler_resolution,
            scales=pooler_scales,
            sampling_ratio=sampling_ratio,
            pooler_type=pooler_type,
        )
        box_head = build_box_head(
            cfg,
            ShapeSpec(
                channels=in_channels, height=pooler_resolution, width=pooler_resolution
            ),
        )
        if cfg.MODEL.ROI_HEADS.LOSS == "CrossEntropy":
            box_predictor = FastRCNNOutputLayers(cfg, box_head.output_shape)
        elif cfg.MODEL.ROI_HEADS.LOSS == "FocalLoss":
            box_predictor = FastRCNNFocaltLossOutputLayers(cfg, box_head.output_shape)
        else:
            raise ValueError("Unknown ROI head loss.")
        # weights = cfg.MODEL.ROI_BOX_HEAD.BBOX_REG_WEIGHTS
        return {
            "box_in_features": in_features,
            "box_pooler": box_pooler,
            "box_head": box_head,
            "box_predictor": box_predictor,
        }

    def forward(
        self,
        images: ImageList,
        features: Dict[str, torch.Tensor],
        proposals: List[Instances],
        targets: Optional[List[Instances]] = None,
        compute_loss=True,
        branch="",
        guide:str='',
        weight=torch.tensor(1.0),
        compute_val_loss=False,
    ) -> Tuple[List[Instances], Dict[str, torch.Tensor]]:

        del images
        if self.training and compute_loss:  # apply if training loss
            assert targets
            # 1000 --> 512
            proposals = self.label_and_sample_proposals(
                proposals, targets, branch=branch
            )
        elif compute_val_loss:  # apply if val loss
            assert targets
            # 1000 --> 512
            temp_proposal_append_gt = self.proposal_append_gt
            self.proposal_append_gt = False
            proposals = self.label_and_sample_proposals(
                proposals, targets, branch=branch
            )  # do not apply target on proposals
            self.proposal_append_gt = temp_proposal_append_gt
        del targets

        if (self.training and compute_loss) or compute_val_loss:
            losses, _ = self._forward_box(
                features, proposals, compute_loss, compute_val_loss, branch,guide
            )
            return proposals, losses
        else:
            pred_instances, predictions = self._forward_box(
                features, proposals, compute_loss, compute_val_loss, branch,guide
            )

            return pred_instances, predictions

    def fast_rcnn_inference(
            self,
            boxes: List[torch.Tensor],
            scores: List[torch.Tensor],
            image_shapes: List[Tuple[int, int]],
            score_thresh: float,
            nms_thresh: float,
            topk_per_image: int,
    ):
        """
        Call `fast_rcnn_inference_single_image` for all images.

        Args:
            boxes (list[Tensor]): A list of Tensors of predicted class-specific or class-agnostic
                boxes for each image. Element i has shape (Ri, K * 4) if doing
                class-specific regression, or (Ri, 4) if doing class-agnostic
                regression, where Ri is the number of predicted objects for image i.
                This is compatible with the output of :meth:`FastRCNNOutputLayers.predict_boxes`.
            scores (list[Tensor]): A list of Tensors of predicted class scores for each image.
                Element i has shape (Ri, K + 1), where Ri is the number of predicted objects
                for image i. Compatible with the output of :meth:`FastRCNNOutputLayers.predict_probs`.
            image_shapes (list[tuple]): A list of (width, height) tuples for each image in the batch.
            score_thresh (float): Only return detections with a confidence score exceeding this
                threshold.
            nms_thresh (float):  The threshold to use for box non-maximum suppression. Value in [0, 1].
            topk_per_image (int): The number of top scoring detections to return. Set < 0 to return
                all detections.

        Returns:
            instances: (list[Instances]): A list of N instances, one for each image in the batch,
                that stores the topk most confidence detections.
            kept_indices: (list[Tensor]): A list of 1D tensor of length of N, each element indicates
                the corresponding boxes/scores index in [0, Ri) from the input, for image i.
        """
        result_per_image = [
            self.fast_rcnn_inference_single_image(
                boxes_per_image, scores_per_image, image_shape, score_thresh, nms_thresh, topk_per_image,wbf=False
            )
            for scores_per_image, boxes_per_image, image_shape in zip(scores, boxes, image_shapes)
        ]
        return [x[0] for x in result_per_image], [x[1] for x in result_per_image]


    def fast_rcnn_inference_single_image(
            self,
            boxes,
            scores,
            image_shape: Tuple[int, int],
            score_thresh: float,
            nms_thresh: float,
            topk_per_image: int,
            wbf = False,
    ):
        """
        Single-image inference. Return bounding-box detection results by thresholding
        on scores and applying non-maximum suppression (NMS).

        Args:
            Same as `fast_rcnn_inference`, but with boxes, scores, and image shapes
            per image.

        Returns:
            Same as `fast_rcnn_inference`, but for only one image.
        """
        if wbf == True:
            boxes_list = [boxes]
            scores_list = [scores[:, 0]]
            labels_list = [torch.zeros_like(scores[:, 0])]

            iou_thr = 0.5
            skip_box_thr = 0.0001
            new_boxes, new_scores, new_labels = weighted_boxes_fusion(boxes_list, scores_list, labels_list,
                                                                      iou_thr=iou_thr, skip_box_thr=skip_box_thr)
            # valid = new_scores>=0.5
            # boxes = Boxes(torch.tensor(new_boxes[valid]))
            # boxes.clip(image_shape)
            # result = Instances(image_shape)
            # result.pred_boxes = boxes
            # result.scores = torch.tensor(new_scores[valid])
            # result.pred_classes = torch.tensor(new_labels[valid])


            boxes = Boxes(torch.tensor(new_boxes))
            boxes.clip(image_shape)
            result = Instances(image_shape)
            result.pred_boxes = boxes
            result.scores = torch.tensor(new_scores)
            result.pred_classes = torch.tensor(new_labels)
            return result, new_labels
        else:

            valid_mask = torch.isfinite(boxes).all(dim=1) & torch.isfinite(scores).all(dim=1)
            if not valid_mask.all():
                boxes = boxes[valid_mask]
                scores = scores[valid_mask]

            scores = scores[:, :-1]
            num_bbox_reg_classes = boxes.shape[1] // 4
            # Convert to Boxes to use the `clip` function ...
            boxes = Boxes(boxes.reshape(-1, 4))
            boxes.clip(image_shape)
            boxes = boxes.tensor.view(-1, num_bbox_reg_classes, 4)  # R x C x 4

            # 1. Filter results based on detection scores. It can make NMS more efficient
            #    by filtering out low-confidence detections.
            filter_mask = scores > score_thresh  # R x K
            # R' x 2. First column contains indices of the R predictions;
            # Second column contains indices of classes.
            filter_inds = filter_mask.nonzero()
            if num_bbox_reg_classes == 1:
                boxes = boxes[filter_inds[:, 0], 0]
            else:
                boxes = boxes[filter_mask]
            scores = scores[filter_mask]

            # 2. Apply NMS for each class independently.
            keep = batched_nms(boxes, scores, filter_inds[:, 1], nms_thresh)
            if topk_per_image >= 0:
                keep = keep[:topk_per_image]
            boxes, scores, filter_inds = boxes[keep], scores[keep], filter_inds[keep]
            #阈值筛选
            # valid = scores >= 0.7
            # result = Instances(image_shape)
            # result.pred_boxes = Boxes(boxes[valid])
            # result.scores = scores[valid]
            # result.pred_classes = filter_inds[:, 1][valid]


            result = Instances(image_shape)
            result.pred_boxes = Boxes(boxes)
            result.scores = scores
            result.pred_classes = filter_inds[:, 1]

            return result, filter_inds[:, 0]

    def predict_boxes(
            self, predictions: Tuple[torch.Tensor, torch.Tensor], proposals: List[Instances]
    ):
        """
        Args:
            predictions: return values of :meth:`forward()`.
            proposals (list[Instances]): proposals that match the features that were
                used to compute predictions. The ``proposal_boxes`` field is expected.

        Returns:
            list[Tensor]:
                A list of Tensors of predicted class-specific or class-agnostic boxes
                for each image. Element i has shape (Ri, K * B) or (Ri, B), where Ri is
                the number of proposals for image i and B is the box dimension (4 or 5)
        """
        if not len(proposals):
            return []
        _, proposal_deltas = predictions
        num_prop_per_image = [len(p) for p in proposals]
        proposal_boxes = [p.proposal_boxes for p in proposals]
        proposal_boxes = proposal_boxes[0].cat(proposal_boxes).tensor


        predict_boxes = Box2BoxTransform(weights=(10.0,10.0,5.0,5.0)).apply_deltas(
            # ensure fp32 for decoding precision
            deltas=proposal_deltas,
            boxes=proposal_boxes,
        )  # Nx(KxB)
        return predict_boxes.split(num_prop_per_image)

    def _predict_boxes(self):
        """
        Returns:
            Tensor: A Tensors of predicted class-specific or class-agnostic boxes
                for all images in a batch. Element i has shape (Ri, K * B) or (Ri, B), where Ri is
                the number of predicted objects for image i and B is the box dimension (4 or 5)
        """
        return FastRCNNOutputLayers.box2box_transform.apply_deltas(self.pred_proposal_deltas, self.proposals.tensor)

    def predict_probs(
            self, predictions: Tuple[torch.Tensor, torch.Tensor], proposals: List[Instances]
    ):
        """
        Args:
            predictions: return values of :meth:`forward()`.
            proposals (list[Instances]): proposals that match the features that were
                used to compute predictions.

        Returns:
            list[Tensor]:
                A list of Tensors of predicted class probabilities for each image.
                Element i has shape (Ri, K + 1), where Ri is the number of proposals for image i.
        """
        scores, _ = predictions
        num_inst_per_image = [len(p) for p in proposals]
        probs = F.softmax(scores, dim=-1)
        return probs.split(num_inst_per_image, dim=0)

    def inference(self, predictions: Tuple[torch.Tensor, torch.Tensor], proposals: List[Instances]):
        """
        Args:
            predictions: return values of :meth:`forward()`.
            proposals (list[Instances]): proposals that match the features that were
                used to compute predictions. The ``proposal_boxes`` field is expected.

        Returns:
            list[Instances]: same as `fast_rcnn_inference`.
            list[Tensor]: same as `fast_rcnn_inference`.
        """
        boxes = self.predict_boxes(predictions, proposals)
        scores = self.predict_probs(predictions, proposals)
        image_shapes = [x.image_size for x in proposals]
        setattr(self,'test_score_thresh',0.0001)
        setattr(self,'test_nms_thresh',0.5)
        setattr(self,'test_topk_per_image',100)
        return self.fast_rcnn_inference(
            boxes,
            scores,
            image_shapes,
            self.test_score_thresh,
            self.test_nms_thresh,
            self.test_topk_per_image,
        )

    def _forward_box(
        self,
        features: Dict[str, torch.Tensor],
        # ds_feat:[torch.tensor],

        proposals: List[Instances],
        compute_loss: bool = True,
        compute_val_loss: bool = False,
        branch: str = "",
        guide: str = ""
    ) -> Union[Dict[str, torch.Tensor], List[Instances]]:
        # if not self.training and compute_loss:
        #     features = [ds_feat[f] for f in self.box_in_features]
        #     box_features = self.box_pooler(features, [x.proposal_boxes for x in proposals])
        # else:
        if guide == 'source':
            box_features = self.box_pooler([features], [x.proposal_boxes for x in proposals])
        else:
            features = [features[f] for f in self.box_in_features]
            box_features = self.box_pooler(features, [x.proposal_boxes for x in proposals])
        box_features = self.box_head(box_features)
        predictions = self.box_predictor(box_features)  #1000
        del box_features

        if (
            self.training and compute_loss
        ) or compute_val_loss:  # apply if training loss or val loss
            losses = self.box_predictor.losses(predictions, proposals)

            if self.train_on_pred_boxes:
                with torch.no_grad():
                    pred_boxes = self.box_predictor.predict_boxes_for_gt_classes(
                        predictions, proposals
                    )
                    for proposals_per_image, pred_boxes_per_image in zip(
                        proposals, pred_boxes
                    ):
                        proposals_per_image.proposal_boxes = Boxes(pred_boxes_per_image)
            return losses, predictions
        else:
#handel head infer
            # pred_instances, _ = self.inference(predictions, proposals)
            #
            pred_instances, _ = self.box_predictor.inference(predictions, proposals)
            return pred_instances, predictions

    @torch.no_grad()
    def label_and_sample_proposals(
        self, proposals: List[Instances], targets: List[Instances], branch: str = ""
    ) -> List[Instances]:
        gt_boxes = [x.gt_boxes for x in targets]
        if self.proposal_append_gt:
            proposals = add_ground_truth_to_proposals(gt_boxes, proposals)

        proposals_with_gt = []

        num_fg_samples = []
        num_bg_samples = []
        for proposals_per_image, targets_per_image in zip(proposals, targets):
            has_gt = len(targets_per_image) > 0
            match_quality_matrix = pairwise_iou(
                targets_per_image.gt_boxes, proposals_per_image.proposal_boxes
            )
            matched_idxs, matched_labels = self.proposal_matcher(match_quality_matrix)
            sampled_idxs, gt_classes = self._sample_proposals(
                matched_idxs, matched_labels, targets_per_image.gt_classes
            )

            proposals_per_image = proposals_per_image[sampled_idxs]
            proposals_per_image.gt_classes = gt_classes

            if has_gt:
                sampled_targets = matched_idxs[sampled_idxs]
                for (trg_name, trg_value) in targets_per_image.get_fields().items():
                    if trg_name.startswith("gt_") and not proposals_per_image.has(
                        trg_name
                    ):
                        proposals_per_image.set(trg_name, trg_value[sampled_targets])
            else:
                gt_boxes = Boxes(
                    targets_per_image.gt_boxes.tensor.new_zeros((len(sampled_idxs), 4))
                )
                proposals_per_image.gt_boxes = gt_boxes

            num_bg_samples.append((gt_classes == self.num_classes).sum().item())
            num_fg_samples.append(gt_classes.numel() - num_bg_samples[-1])
            proposals_with_gt.append(proposals_per_image)

        storage = get_event_storage()
        storage.put_scalar(
            "roi_head/num_target_fg_samples_" + branch, np.mean(num_fg_samples)
        )
        storage.put_scalar(
            "roi_head/num_target_bg_samples_" + branch, np.mean(num_bg_samples)
        )

        return proposals_with_gt
