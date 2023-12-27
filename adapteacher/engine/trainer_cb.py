#
# import os
# import time
# import logging
# import torch
# import matplotlib
# matplotlib.use('Agg')
# import matplotlib.pyplot as plt
# import torch.nn as nn
# import torch.nn.functional as F
# from torch.nn.parallel import DistributedDataParallel
# from fvcore.nn.precise_bn import get_bn_modules
# import numpy as np
# from collections import OrderedDict
# # from matplotlib.font_manager import FontProperties
# # import cv2
# # import numpy as np
# import detectron2.utils.comm as comm
# from detectron2.checkpoint import DetectionCheckpointer
# from detectron2.engine import DefaultTrainer, SimpleTrainer, TrainerBase
# from detectron2.engine.train_loop import AMPTrainer
# from detectron2.utils.events import EventStorage
# from detectron2.evaluation import verify_results, DatasetEvaluators
# # from detectron2.evaluation import COCOEvaluator, verify_results, DatasetEvaluators
#
# from detectron2.data.dataset_mapper import DatasetMapper
# from detectron2.engine import hooks
# from detectron2.structures.boxes import Boxes
# from detectron2.structures.instances import Instances
# from detectron2.utils.env import TORCH_VERSION
# from detectron2.data import MetadataCatalog
#
# from adapteacher.data.build import (
#     build_detection_semisup_train_loader,
#     build_detection_test_loader,
#     build_detection_semisup_train_loader_two_crops,
# )
# from adapteacher.data.dataset_mapper import DatasetMapperTwoCropSeparate
# from adapteacher.engine.hooks import LossEvalHook
# # from adapteacher.modeling.meta_arch.ts_ensemble import EnsembleTSModel
# from adapteacher.checkpoint.detection_checkpoint import DetectionTSCheckpointer
# from adapteacher.solver.build import build_lr_scheduler
# from adapteacher.evaluation import PascalVOCDetectionEvaluator, COCOEvaluator
# # from utils.fft import FDA_source_to_target_np
# from .probe import OpenMatchTrainerProbe
# import copy
# import collections
# from torch.utils.tensorboard import SummaryWriter
#
#
# # Supervised-only Trainer
# class BaselineTrainer(DefaultTrainer):
#     def __init__(self, cfg):
#         """
#         Args:
#             cfg (CfgNode):
#         Use the custom checkpointer, which loads other backbone models
#         with matching heuristics.
#         """
#         cfg = DefaultTrainer.auto_scale_workers(cfg, comm.get_world_size())
#         model = self.build_model(cfg)
#         optimizer = self.build_optimizer(cfg, model)
#         data_loader = self.build_train_loader(cfg)
#
#         if comm.get_world_size() > 1:
#             model = DistributedDataParallel(
#                 model, device_ids=[comm.get_local_rank()], broadcast_buffers=False
#             )
#
#         TrainerBase.__init__(self)
#         self._trainer = (AMPTrainer if cfg.SOLVER.AMP.ENABLED else SimpleTrainer)(
#             model, data_loader, optimizer
#         )
#
#         self.scheduler = self.build_lr_scheduler(cfg, optimizer)
#         self.checkpointer = DetectionCheckpointer(
#             model,
#             cfg.OUTPUT_DIR,
#             optimizer=optimizer,
#             scheduler=self.scheduler,
#         )
#         self.start_iter = 0
#         self.start_round = 0
#         self.max_iter = cfg.SOLVER.MAX_ITER
#         self.max_round = cfg.SOLVER.MAX_ROUND
#         self.cfg = cfg
#
#         self.register_hooks(self.build_hooks())
#
#     def resume_or_load(self, resume=True):
#         """
#         If `resume==True` and `cfg.OUTPUT_DIR` contains the last checkpoint (defined by
#         a `last_checkpoint` file), resume from the file. Resuming means loading all
#         available states (eg. optimizer and scheduler) and update iteration counter
#         from the checkpoint. ``cfg.MODEL.WEIGHTS`` will not be used.
#         Otherwise, this is considered as an independent training. The method will load model
#         weights from the file `cfg.MODEL.WEIGHTS` (but will not load other states) and start
#         from iteration 0.
#         Args:
#             resume (bool): whether to do resume or not
#         """
#         checkpoint = self.checkpointer.resume_or_load(
#             self.cfg.MODEL.WEIGHTS, resume=resume
#         )
#         if resume and self.checkpointer.has_checkpoint():
#             self.start_iter = checkpoint.get("iteration", -1) + 1
#             # The checkpoint stores the training iteration that just finished, thus we start
#             # at the next iteration (or iter zero if there's no checkpoint).
#         if isinstance(self.model, DistributedDataParallel):
#             # broadcast loaded data/model from the first rank, because other
#             # machines may not have access to the checkpoint file
#             if TORCH_VERSION >= (1, 7):
#                 self.model._sync_params_and_buffers()
#             self.start_iter = comm.all_gather(self.start_iter)[0]
#
#     def train_loop(self, start_iter: int, max_iter: int,start_round: int,max_round:int ):
#         """
#         Args:
#             start_iter, max_iter (int): See docs above
#         """
#         logger = logging.getLogger(__name__)
#         logger.info("Starting training from iteration {}".format(start_iter))
#         self.round = self.start_round = start_round
#         self.iter = self.start_iter = start_iter
#         self.max_iter = max_iter
#         self.max_round = max_round
#
#         with EventStorage(start_iter) as self.storage:
#             try:
#                 self.before_train()
#                 for self.iter in range(start_iter, max_iter):
#                     self.before_step()
#                     self.run_step()
#                     self.after_step()
#             except Exception:
#                 logger.exception("Exception during training:")
#                 raise
#             finally:
#                 self.after_train()
#
#     def run_step(self):
#         self._trainer.iter = self.iter
#
#         assert self.model.training, "[SimpleTrainer] model was changed to eval mode!"
#         start = time.perf_counter()
#
#         data = next(self._trainer._data_loader_iter)
#         data_time = time.perf_counter() - start
#
#         record_dict, _, _, _ = self.model(data, branch="supervised")
#
#         num_gt_bbox = 0.0
#         for element in data:
#             num_gt_bbox += len(element["instances"])
#         num_gt_bbox = num_gt_bbox / len(data)
#         record_dict["bbox_num/gt_bboxes"] = num_gt_bbox
#
#         loss_dict = {}
#         for key in record_dict.keys():
#             if key[:4] == "loss" and key[-3:] != "val":
#                 loss_dict[key] = record_dict[key]
#
#         losses = sum(loss_dict.values())
#
#         metrics_dict = record_dict
#         metrics_dict["data_time"] = data_time
#         self._write_metrics(metrics_dict)
#
#         self.optimizer.zero_grad()
#         losses.backward()
#         self.optimizer.step()
#
#     @classmethod
#     def build_evaluator(cls, cfg, dataset_name, output_folder=None):
#         if output_folder is None:
#             output_folder = os.path.join(cfg.OUTPUT_DIR, "inference")
#         evaluator_list = []
#         evaluator_type = MetadataCatalog.get(dataset_name).evaluator_type
#
#         if evaluator_type == "coco":
#             evaluator_list.append(COCOEvaluator(
#                 dataset_name, output_dir=output_folder))
#         elif evaluator_type == "pascal_voc":
#             return PascalVOCDetectionEvaluator(dataset_name)
#         elif evaluator_type == "pascal_voc_water":
#             return PascalVOCDetectionEvaluator(dataset_name,
#                                                target_classnames=["bicycle", "bird", "car", "cat", "dog", "person"])
#         if len(evaluator_list) == 0:
#             raise NotImplementedError(
#                 "no Evaluator for the dataset {} with the type {}".format(
#                     dataset_name, evaluator_type
#                 )
#             )
#         elif len(evaluator_list) == 1:
#             return evaluator_list[0]
#
#         return DatasetEvaluators(evaluator_list)
#
#     @classmethod
#     def build_train_loader(cls, cfg):
#         return build_detection_semisup_train_loader(cfg, mapper=None)
#
#     @classmethod
#     def build_test_loader(cls, cfg, dataset_name):
#         """
#         Returns:
#             iterable
#         """
#         return build_detection_test_loader(cfg, dataset_name)
#
#     def build_hooks(self):
#         """
#         Build a list of default hooks, including timing, evaluation,
#         checkpointing, lr scheduling, precise BN, writing events.
#
#         Returns:
#             list[HookBase]:
#         """
#         cfg = self.cfg.clone()
#         cfg.defrost()
#         cfg.DATALOADER.NUM_WORKERS = 0
#
#         ret = [
#             hooks.IterationTimer(),
#             hooks.LRScheduler(self.optimizer, self.scheduler),
#             hooks.PreciseBN(
#                 cfg.TEST.EVAL_PERIOD,
#                 self.model,
#                 self.build_train_loader(cfg),
#                 cfg.TEST.PRECISE_BN.NUM_ITER,
#             )
#             if cfg.TEST.PRECISE_BN.ENABLED and get_bn_modules(self.model)
#             else None,
#         ]
#
#         if comm.is_main_process():
#             ret.append(
#                 hooks.PeriodicCheckpointer(
#                     self.checkpointer, cfg.SOLVER.CHECKPOINT_PERIOD
#                 )
#             )
#
#         def test_and_save_results():
#             self._last_eval_results = self.test(self.cfg, self.model)
#             return self._last_eval_results
#
#         ret.append(hooks.EvalHook(cfg.TEST.EVAL_PERIOD, test_and_save_results))
#
#         if comm.is_main_process():
#             ret.append(hooks.PeriodicWriter(self.build_writers(), period=20))
#         return ret
#
#     def _write_metrics(self, metrics_dict: dict):
#         """
#         Args:
#             metrics_dict (dict): dict of scalar metrics
#         """
#         metrics_dict = {
#             k: v.detach().cpu().item() if isinstance(v, torch.Tensor) else float(v)
#             for k, v in metrics_dict.items()
#         }
#         # gather metrics among all workers for logging
#         # This assumes we do DDP-style training, which is currently the only
#         # supported method in detectron2.
#         all_metrics_dict = comm.gather(metrics_dict)
#
#         if comm.is_main_process():
#             if "data_time" in all_metrics_dict[0]:
#                 data_time = np.max([x.pop("data_time")
#                                     for x in all_metrics_dict])
#                 self.storage.put_scalar("data_time", data_time)
#
#             metrics_dict = {
#                 k: np.mean([x[k] for x in all_metrics_dict])
#                 for k in all_metrics_dict[0].keys()
#             }
#
#             loss_dict = {}
#             for key in metrics_dict.keys():
#                 if key[:4] == "loss":
#                     loss_dict[key] = metrics_dict[key]
#
#             total_losses_reduced = sum(loss for loss in loss_dict.values())
#
#             self.storage.put_scalar("total_loss", total_losses_reduced)
#             if len(metrics_dict) > 1:
#                 self.storage.put_scalars(**metrics_dict)
#
# def bb_intersection_over_union(A, B):
#     xA = max(A[0], B[0])
#     yA = max(A[1], B[1])
#     xB = min(A[2], B[2])
#     yB = min(A[3], B[3])
#
#     # compute the area of intersection rectangle
#     interArea = max(0, xB - xA) * max(0, yB - yA)
#
#     if interArea == 0:
#         return 0.0
#
#     # compute the area of both the prediction and ground-truth rectangles
#     boxAArea = (A[2] - A[0]) * (A[3] - A[1])
#     boxBArea = (B[2] - B[0]) * (B[3] - B[1])
#
#     iou = interArea / float(boxAArea + boxBArea - interArea)
#     return iou
#
#
# def prefilter_boxes(boxes, scores, labels, weights, thr):
#     # Create dict with boxes stored by its label
#     new_boxes = dict()
#     for t in range(len(boxes)):
#         for j in range(len(boxes[t])):
#             score = scores[t][j]
#             if score < thr:
#                 continue
#             label = int(labels[t][j])
#             box_part = boxes[t][j]
#             # box_area = (box_part[3]-box_part[1]) * (box_part[2]-box_part[0])
#             b = [int(label), float(score) * weights[t], float(box_part[0]), float(box_part[1]), float(box_part[2]),
#                  float(box_part[3])]
#             if label not in new_boxes:
#                 new_boxes[label] = []
#             new_boxes[label].append(b)
#
#     # Sort each list in dict by score and transform it to numpy array
#     for k in new_boxes:
#         current_boxes = np.array(new_boxes[k])
#         new_boxes[k] = current_boxes[current_boxes[:, 1].argsort()[::-1]]
#
#     return new_boxes
#
# def get_mIOU(pred_ins, gt_ins, iou_threh = 0.5):
#     pred_boxs, gt_boxs = pred_ins.gt_boxes.tensor, gt_ins.gt_boxes.tensor
#     iou = np.zeros((len(pred_boxs), len(gt_boxs)))
#     #calculate iou for filter
#     for j in range(len(gt_boxs)):
#         for i in range(len(pred_boxs)):
#             iou[i][j] = bb_intersection_over_union(pred_boxs[i],gt_boxs[j])
#     #filter
#     iou_list = []
#     gt_index = iou.argmax(axis=1)
#     if len(pred_boxs)>0:
#         pred_index = iou.argmax(axis=0)
#         for j in range(len(gt_boxs)):
#             iou_list.append(iou[pred_index][j])
#     gt_index[iou.max(axis=1) < iou_threh] = -1
#     select = np.zeros(gt_boxs.size(0), dtype=bool)
#     match = []
#     confidence = []
#     for sid, gt_idx in enumerate(gt_index):
#         if gt_idx >= 0:
#             if not select[gt_idx]:
#                 match.append(1)
#             else:
#                 match.append(0)
#             select[gt_idx] = True
#         else:
#             match.append(0)
#         # iou_list.append(iou[sid][gt_idx])
#         confidence.append(pred_ins.scores[sid].detach().cpu())
#
#     match = np.asarray(match)
#     confidence = np.asarray(confidence)
#     iou_list = np.asarray(iou_list)
#     order = confidence.argsort()[::-1]
#     match = match[order]
#
#     # tp = np.cumsum(match == 1).sum()
#     # fp = np.cumsum(match == 0).sum()
#     tp = (match == 1).sum()
#     fp = (match == 0).sum()
#
#     if len(match) != 0 :
#
#         rec = tp / (fp + tp)
#         prec = tp / len(match)
#         mIOU = iou_list.mean()
#     else:
#         rec = 0.0
#         prec = 0.0
#         mIOU = 0.0
# #这个iou是以gt为基准，原本的是按照pred_box，如果是按面积计，他的无疑更合理
#     return rec, prec, mIOU
#
# def get_mIOU2(pred_ins, gt_ins, iou_threh = 0.5):
#     pred_boxs, gt_boxs = pred_ins.gt_boxes.tensor, gt_ins.gt_boxes.tensor
#     iou = np.zeros((len(pred_boxs), len(gt_boxs)))
#     #calculate iou for filter
#     for j in range(len(gt_boxs)):
#         for i in range(len(pred_boxs)):
#             iou[i][j] = bb_intersection_over_union(pred_boxs[i],gt_boxs[j])
#     #filter
#     iou_list = []
#     gt_index = iou.argmax(axis=1)
#     if len(pred_boxs)>0:
#         pred_index = iou.argmax(axis=0)
#         # for j in range(len(gt_boxs)):
#         #     iou_list.append(iou[pred_index][j])
#     gt_index[iou.max(axis=1) < iou_threh] = -1
#     select = np.zeros(gt_boxs.size(0), dtype=bool)
#     match = []
#     confidence = []
#     for sid, gt_idx in enumerate(gt_index):
#         if gt_idx >= 0:
#             if not select[gt_idx]:
#                 match.append(1)
#                 iou_list.append(iou[sid][gt_idx])
#             else:
#                 match.append(0)
#             select[gt_idx] = True
#         else:
#             match.append(0)
#
#         confidence.append(pred_ins.scores[sid].detach().cpu())
#
#     match = np.asarray(match)
#     confidence = np.asarray(confidence)
#     iou_list = np.asarray(iou_list)
#     order = confidence.argsort()[::-1]
#     match = match[order]
#
#     # tp = np.cumsum(match == 1).sum()
#     # fp = np.cumsum(match == 0).sum()
#     tp = (match == 1).sum()
#     fp = (match == 0).sum()
#
#     rec = tp / (fp + tp)
#     prec = tp / len(match)
#     mIOU = iou_list.mean()
# #这个是按面积计miou
#     return rec, prec, mIOU
#
# def get_weighted_box(boxes, conf_type='avg'):
#     """
#     Create weighted box for set of boxes
#     :param boxes: set of boxes to fuse
#     :param conf_type: type of confidence one of 'avg' or 'max'
#     :return: weighted box
#     """
#     #oral
#     # box = np.zeros(6, dtype=np.float32)
#     # conf = 0
#     # conf_list = []
#     # for b in boxes:
#     #     box[2:] += (b[1] * b[2:])
#     #     conf += b[1]
#     #     conf_list.append(b[1])
#     # box[0] = boxes[0][0]
#     # if conf_type == 'avg':
#     #     box[1] = conf / len(boxes)
#     # elif conf_type == 'max':
#     #     box[1] = np.array(conf_list).max()
#     # box[2:] /= conf
#
#     #area_weights
#     box = np.zeros(6, dtype=np.float32)
#     conf = 0
#     area = 0
#     i = 0
#     box_area1 = int((boxes[0][5] - boxes[0][3]) * (boxes[0][4] - boxes[0][2]))
#     box_area2 = int((boxes[1][5] - boxes[1][3]) * (boxes[1][4] - boxes[1][2]))
#     # area_weights = [box_area1 / box_area2,1]
#     conf_list = []
#     for b in boxes:
#         # box_area = (b[5] -b[3])* (b[4]-b[2])
#         box[2:] += (b[1] * b[2:])
#         conf += b[1]
#         i+=1
#         conf_list.append(b[1])
#     box[0] = boxes[0][0]
#     if conf_type == 'avg':
#         box[1] = conf / len(boxes)
#     elif conf_type == 'max':
#         box[1] = np.array(conf_list).max()
#     box[2:] /= (conf )
#     return box
#
#
# grads = {}
#
#
# # 这个函数是为了获取中间变量的梯度，我方案中的Z不是一个叶子结点，所以其梯度在反向传播之后不会被保存
# def save_grad(name):
#     def hook(grad):
#         grads[name] = grad
#
#     return hook
#
#
# def find_matching_box(boxes_list, new_box, match_iou):
#     best_iou = match_iou
#     best_index = -1
#     for i in range(len(boxes_list)):
#         box = boxes_list[i]
#         if box[0] != new_box[0]:
#             continue
#         iou = bb_intersection_over_union(box[2:], new_box[2:])
#         if iou > best_iou:
#             best_index = i
#             best_iou = iou
#
#     return best_index, best_iou
#
#
# def weighted_boxes_fusion(boxes_list, scores_list, labels_list, weights=None, iou_thr=0.55, skip_box_thr=0.8,
#                           conf_type='avg', allows_overflow=False):
#     '''
#     :param boxes_list: list of boxes predictions from each model, each box is 4 numbers.
#     It has 3 dimensions (models_number, model_preds, 4)
#     Order of boxes: x1, y1, x2, y2. We expect float normalized coordinates [0; 1]
#     :param scores_list: list of scores for each model
#     :param labels_list: list of labels for each model
#     :param weights: list of weights for each model. Default: None, which means weight == 1 for each model
#     :param iou_thr: IoU value for boxes to be a match
#     :param skip_box_thr: exclude boxes with score lower than this variable
#     :param conf_type: how to calculate confidence in weighted boxes. 'avg': average value, 'max': maximum value
#     :param allows_overflow: false if we want confidence score not exceed 1.0
#
#     :return: boxes: boxes coordinates (Order of boxes: x1, y1, x2, y2).
#     :return: scores: confidence scores
#     :return: labels: boxes labels
#     '''
#
#     if weights is None:
#         weights = np.ones(len(boxes_list))
#     if len(weights) != len(boxes_list):
#         print('Warning: incorrect number of weights {}. Must be: {}. Set weights equal to 1.'.format(len(weights),
#                                                                                                      len(boxes_list)))
#         weights = np.ones(len(boxes_list))
#     weights = np.array(weights)
#
#     if conf_type not in ['avg', 'max']:
#         print('Unknown conf_type: {}. Must be "avg" or "max"'.format(conf_type))
#         exit()
#         #filter boxes which score>thr
#     filtered_boxes = prefilter_boxes(boxes_list, scores_list, labels_list, weights, skip_box_thr)
#     if len(filtered_boxes) == 0:
#         return np.zeros((0, 4)), np.zeros((0,)), np.zeros((0,))
#
#     overall_boxes = []
#     for label in filtered_boxes:
#         boxes = filtered_boxes[label]
#         new_boxes = []
#         weighted_boxes = []
#
#         # Clusterize boxes
#         for j in range(0, len(boxes)):
#             index, best_iou = find_matching_box(weighted_boxes, boxes[j], iou_thr)
#             if index != -1:
#                 new_boxes[index].append(boxes[j])
#                 weighted_boxes[index] = get_weighted_box(new_boxes[index], conf_type)
#             else:
#                 new_boxes.append([boxes[j].copy()])
#                 weighted_boxes.append(boxes[j].copy())
#
#         # Rescale confidence based on number of models and boxes
#         for i in range(len(new_boxes)):
#             if not allows_overflow:
#                 weighted_boxes[i][1] = weighted_boxes[i][1] * min(weights.sum(), len(new_boxes[i])) / weights.sum()
#             else:
#                 weighted_boxes[i][1] = weighted_boxes[i][1] * len(new_boxes[i]) / weights.sum()
#         overall_boxes.append(np.array(weighted_boxes))
#
#     overall_boxes = np.concatenate(overall_boxes, axis=0)
#     overall_boxes = overall_boxes[overall_boxes[:, 1].argsort()[::-1]]
#     boxes = overall_boxes[:, 2:]
#     scores = overall_boxes[:, 1]
#     labels = overall_boxes[:, 0]
#     return boxes, scores, labels
#
# # Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved
# from torch.nn.parallel import DataParallel, DistributedDataParallel
# import torch.nn as nn
#
#
# class EnsembleTSModel(nn.Module):
#     def __init__(self, modelTeacher, modelStudent):
#         super(EnsembleTSModel, self).__init__()
#
#         if isinstance(modelTeacher, (DistributedDataParallel, DataParallel)):
#             modelTeacher_ds = modelTeacher.module
#         # if isinstance(modelTeacher_dp, (DistributedDataParallel, DataParallel)):
#         #     modelTeacher_dp = modelTeacher_dp.module
#         if isinstance(modelStudent, (DistributedDataParallel, DataParallel)):
#             modelStudent = modelStudent.module
#         # if isinstance(modelDomain, (DistributedDataParallel, DataParallel)):
#         #     modelStudent = modelDomain.module
#
#         self.modelTeacher = modelTeacher
#         # self.modelTeacher_dp = modelTeacher_dp
#         self.modelStudent = modelStudent
#         # self.modelDomain = modelDomain
#
# # Adaptive Teacher Trainer
# class ATeacherTrainer(DefaultTrainer):
#     def __init__(self, cfg):
#         """
#         Args:
#             cfg (CfgNode):
#         Use the custom checkpointer, which loads other backbone models
#         with matching heuristics.
#         """
#         cfg = DefaultTrainer.auto_scale_workers(cfg, comm.get_world_size())
#         data_loader = self.build_train_loader(cfg)
#
#         # create an student model
#         model = self.build_model(cfg)
#         optimizer = self.build_optimizer(cfg, model)
#
#         # create an teacher model
#         model_teacher = self.build_model(cfg)
#         self.model_teacher = model_teacher
#
#         # model_teacher = self.build_model(cfg)
#         # self.model_teacher_dp = model_teacher
#
#
#
#         # model_domain = self.build_model(cfg)
#         # self.model_domain = model_domain
#
#         # For training, wrap with DDP. But don't need this for inference.
#         if comm.get_world_size() > 1:
#             model = DistributedDataParallel(
#                 model, device_ids=[comm.get_local_rank()], broadcast_buffers=False
#             )
#
#         TrainerBase.__init__(self)
#         self._trainer = (AMPTrainer if cfg.SOLVER.AMP.ENABLED else SimpleTrainer)(
#             model, data_loader, optimizer
#         )
#         self.scheduler = self.build_lr_scheduler(cfg, optimizer)
#
#         # Ensemble teacher and student model is for model saving and loading
#         ensem_ts_model = EnsembleTSModel(model_teacher, model)
#
#         self.checkpointer = DetectionTSCheckpointer(
#             ensem_ts_model,
#             cfg.OUTPUT_DIR,
#             optimizer=optimizer,
#             scheduler=self.scheduler,
#         )
#         self.start_iter = 0
#         self.start_round = 0
#         self.max_iter = cfg.SOLVER.MAX_ITER
#         self.max_round = cfg.SOLVER.MAX_ROUND
#         self.cfg = cfg
#
#         self.probe = OpenMatchTrainerProbe(cfg)
#         self.register_hooks(self.build_hooks())
#
#     def resume_or_load(self, resume=True):
#         """
#         If `resume==True` and `cfg.OUTPUT_DIR` contains the last checkpoint (defined by
#         a `last_checkpoint` file), resume from the file. Resuming means loading all
#         available states (eg. optimizer and scheduler) and update iteration counter
#         from the checkpoint. ``cfg.MODEL.WEIGHTS`` will not be used.
#         Otherwise, this is considered as an independent training. The method will load model
#         weights from the file `cfg.MODEL.WEIGHTS` (but will not load other states) and start
#         from iteration 0.
#         Args:
#             resume (bool): whether to do resume or not
#         """
#         checkpoint = self.checkpointer.resume_or_load(
#             self.cfg.MODEL.WEIGHTS, resume=resume
#         )
#         # checkpoint_dp = self.checkpointer.resume_or_load(
#         #     self.cfg.MODEL.WEIGHTS_DP, resume=resume
#         # )
#         if resume and self.checkpointer.has_checkpoint():
#             self.start_iter = checkpoint.get("iteration", -1) + 1
#             # self.start_iter = checkpoint_dp.get("iteration", -1) + 1
#             # The checkpoint stores the training iteration that just finished, thus we start
#             # at the next iteration (or iter zero if there's no checkpoint).
#         if isinstance(self.model, DistributedDataParallel):
#             # broadcast loaded data/model from the first rank, because other
#             # machines may not have access to the checkpoint file
#             # if TORCH_VERSION >= (1, 7):
#             #     self.model._sync_params_and_buffers()
#             self.start_iter = comm.all_gather(self.start_iter)[0]
#
#     @classmethod
#     def build_evaluator(cls, cfg, dataset_name, output_folder=None):
#         if output_folder is None:
#             output_folder = os.path.join(cfg.OUTPUT_DIR, "inference")
#         evaluator_list = []
#         evaluator_type = MetadataCatalog.get(dataset_name).evaluator_type
#
#         if evaluator_type == "coco":
#             evaluator_list.append(COCOEvaluator(
#                 dataset_name, output_dir=output_folder))
#         elif evaluator_type == "pascal_voc":
#             return PascalVOCDetectionEvaluator(dataset_name)
#         elif evaluator_type == "pascal_voc_water":
#             return PascalVOCDetectionEvaluator(dataset_name,
#                                                target_classnames=["bicycle", "bird", "car", "cat", "dog", "person"])
#         if len(evaluator_list) == 0:
#             raise NotImplementedError(
#                 "no Evaluator for the dataset {} with the type {}".format(
#                     dataset_name, evaluator_type
#                 )
#             )
#         elif len(evaluator_list) == 1:
#             return evaluator_list[0]
#
#         return DatasetEvaluators(evaluator_list)
#
#     @classmethod
#     def build_train_loader(cls, cfg):
#         mapper = DatasetMapperTwoCropSeparate(cfg, True)
#         return build_detection_semisup_train_loader_two_crops(cfg, mapper)
#
#     @classmethod
#     def build_lr_scheduler(cls, cfg, optimizer):
#         return build_lr_scheduler(cfg, optimizer)
#
#     def train(self,teacher_ds,teacher_dp):
#         self.train_loop(self.start_iter, self.max_iter,self.start_round,self.max_round,teacher_ds,teacher_dp)
#         if hasattr(self, "_last_eval_results") and comm.is_main_process():
#             verify_results(self.cfg, self._last_eval_results)
#             return self._last_eval_results
#
#     def train_loop(self, start_iter: int, max_iter: int,start_round: int, max_round: int,teachers_ds,teacher_dp):
#         logger = logging.getLogger(__name__)
#         logger.info("Starting training from iteration {}".format(start_iter))
#
#         self.iter = self.start_iter = start_iter
#         self.max_iter = max_iter
#         self.round = self.start_round = start_round
#         self.max_round = max_round
#
#         with EventStorage(start_iter) as self.storage:
#             try:
#                 self.before_train()
#                 # recall_list = []
#                 # precise_list = []
#                 # mIOU_list = []
#                 # t_ins_score_list = []
#                 # t_score_list = []
#                 # s_ins_score_list = []
#                 # s_score_list = []
#                 # level_diction_list = dict(p2=[], p3=[], p4=[], p5=[], p6=[])
#                 # source_level_diction_list = dict(p2=[], p3=[], p4=[], p5=[], p6=[])
#                 for self.round in range(start_round,max_round):
#                     for self.iter in range(start_iter, max_iter):
#                     # for self.iter in range(2000, max_iter):
#                         self.before_step()
#                         self.run_step_full_semisup(teachers_ds,teacher_dp)
#                         self.after_step()
#                     start_iter = 0
#
#
#             except Exception:
#                 logger.exception("Exception during training:")
#                 raise
#             finally:
#                 self.after_train()
#
#     def NonMaxSuppression(self, proposal_bbox_inst, confi_thres=0.7,nms_thresh = 0.45, proposal_type="roih"):
#         if proposal_type == "roih":
#             valid_map = proposal_bbox_inst.scores > confi_thres
#
#             # create instances containing boxes and gt_classes
#             image_shape = proposal_bbox_inst.image_size
#             new_proposal_inst = Instances(image_shape)
#
#             # create box  #actually no need valid_map
#             new_bbox_loc = proposal_bbox_inst.pred_boxes.tensor[valid_map, :]
#             new_score = proposal_bbox_inst.scores[valid_map,:]
#             new_class = proposal_bbox_inst.pred_classes[valid_map,:]
#             # new_bbox_loc = proposal_bbox_inst.pred_boxes.tensor
#             # new_score = proposal_bbox_inst.scores
#             # new_class = proposal_bbox_inst.pred_classes
#             scores,index = new_score.sort(descending = True)
#             keep_inds = []
#             while(len(index) > 0):
#                 cur_inx = index[0]
#                 cur_score = scores[cur_inx]
#                 if cur_score < confi_thres:
#                     break;
#                 keep = True
#                 for ind in keep_inds:
#                     current_bbox = new_bbox_loc[cur_inx]
#                     remain_box = new_bbox_loc[ind]
#                     # iou = 1
#                     ioc = self.box_ioc_xyxy(current_bbox,remain_box)
#                     if ioc > nms_thresh:
#                         keep = False
#                         break
#
#                 if keep:
#                     keep_inds.append(cur_inx)
#                 index = index[1:]
#             # if len(keep_inds) == 0:
#             #     valid_map = proposal_bbox_inst.scores > thres
#             #
#             #     # create instances containing boxes and gt_classes
#             #     image_shape = proposal_bbox_inst.image_size
#             #     new_proposal_inst = Instances(image_shape)
#             #
#             #     # create box
#             #     new_bbox_loc = proposal_bbox_inst.pred_boxes.tensor[valid_map, :]
#             #     new_boxes = Boxes(new_bbox_loc)
#             #
#             #     # add boxes to instances
#             #     new_proposal_inst.gt_boxes = new_boxes
#             #     new_proposal_inst.gt_classes = proposal_bbox_inst.pred_classes[valid_map]
#             #     new_proposal_inst.scores = proposal_bbox_inst.scores[valid_map]
#             #     for i in new_proposal_inst.scores:
#             #         i = 0
#             #     return new_proposal_inst
#
#
#
#             keep_inds = torch.tensor(keep_inds)
#             score_nms = new_score[keep_inds.long()]
#             # score_nms = score_nms.reshape(-1,1)
#             # score_nms = score_nms.reshape(-1)
#             box_nms = new_bbox_loc[keep_inds.long()]
#             box_nms = box_nms.reshape(-1,4)
#             box_nms = Boxes(box_nms)
#             class_nms = new_class[keep_inds.long()]
#             # class_nms = class_nms.reshape(-1,1)
#             new_proposal_inst.gt_boxes = box_nms
#             new_proposal_inst.gt_classes = class_nms
#             new_proposal_inst.scores = score_nms
#
#
#         elif proposal_type == "rpn":
#
#             raise ValueError("Unknown NMS branches")
#
#         return new_proposal_inst
#     # =====================================================
#     # ================== Pseduo-labeling ==================
#     # =====================================================
#     def threshold_bbox(self, proposal_bbox_inst, thres=0.7, proposal_type="roih"):
#         if proposal_type == "rpn":
#             valid_map = proposal_bbox_inst.objectness_logits > thres
#
#             # create instances containing boxes and gt_classes
#             image_shape = proposal_bbox_inst.image_size
#             new_proposal_inst = Instances(image_shape)
#
#             # create box
#             new_bbox_loc = proposal_bbox_inst.proposal_boxes.tensor[valid_map, :]
#             new_boxes = Boxes(new_bbox_loc)
#
#             # add boxes to instances
#             new_proposal_inst.gt_boxes = new_boxes
#             new_proposal_inst.objectness_logits = proposal_bbox_inst.objectness_logits[
#                 valid_map
#             ]
#         elif proposal_type == "roih":
#             valid_map = proposal_bbox_inst.scores > thres
#
#             # create instances containing boxes and gt_classes
#             image_shape = proposal_bbox_inst.image_size
#             new_proposal_inst = Instances(image_shape)
#
#             # create box
#             new_bbox_loc = proposal_bbox_inst.pred_boxes.tensor[valid_map, :]
#             new_boxes = Boxes(new_bbox_loc)
#
#             # add boxes to instances
#             new_proposal_inst.gt_boxes = new_boxes
#             new_proposal_inst.gt_classes = proposal_bbox_inst.pred_classes[valid_map]
#             new_proposal_inst.scores = proposal_bbox_inst.scores[valid_map]
#
#         return new_proposal_inst
#
#     def box_ioc_xyxy(self,box1_rank,box2_sus):
#
#         xA = max(box1_rank[0],box2_sus[0])
#         yA = max(box1_rank[1],box2_sus[1])
#         xB = min(box1_rank[2], box2_sus[2])
#         yB = min(box1_rank[3], box2_sus[3])
#
#         intersect = max(0,xB - xA + 1) * max(0,yB - yA + 1)
#         # box_area1 = (box1[2]-box1[0] + 1) * (box1[3] - box1[1] + 1)
#         box_area2 = (box2_sus[2] - box2_sus[0] + 1) * (box2_sus[3] - box2_sus[1] + 1)
#
#         # ioc = intersect / float(box_area2 + box_area1 -intersect)
#         ioc = intersect / float(box_area2)
#         return ioc
#
#     def Knowlegde_Fusion(self,proposals_T, proposals_S, iou_thr=0.5, skip_box_thr=0.05, weights=[1, 1]):
#         assert len(proposals_T) == len(proposals_S)
#         list_instances = []
#         num_proposal_output = 0.0
#         for i in range(len(proposals_T)):
#             pseudo_label_inst = self.pseudo_fusion(proposals_T[i], proposals_S[i], iou_thr, skip_box_thr, weights)
#
#             num_proposal_output += len(pseudo_label_inst)
#             list_instances.append(pseudo_label_inst)
#         num_proposal_output = num_proposal_output / (len(proposals_T) + len(proposals_S))
#         return list_instances, num_proposal_output
#
#     def pseudo_fusion(self,output_t, output_s, iou_thr=0.5, skip_box_thr=0.05, weights=[1, 1]):
#
#         image_size = output_t.image_size
#
#         boxes_list, scores_list, labels_list = [], [], []
#
#         box_list_t = output_t.pred_boxes.tensor
#         scores_list_t = output_t.scores
#         classes_list_t = output_t.pred_classes
#
#         box_list_s = output_s.pred_boxes.tensor
#         scores_list_s = output_s.scores
#         classes_list_s = output_s.pred_classes
#
#         boxes_list.append(box_list_t)
#         boxes_list.append(box_list_s)
#         scores_list.append(scores_list_t)
#         scores_list.append(scores_list_s)
#         labels_list.append(classes_list_t)
#         labels_list.append(classes_list_s)
#         boxes, scores, labels = weighted_boxes_fusion(boxes_list, scores_list, labels_list, weights=weights,
#                                                       iou_thr=iou_thr, skip_box_thr=skip_box_thr)
#         result = Instances(image_size)
#         boxes = Boxes(torch.tensor(boxes).cuda())
#         boxes.clip(image_size)
#         result.gt_boxes = boxes
#         result.scores = torch.tensor(scores).cuda()
#         result.gt_classes = torch.tensor(labels).cuda().long()
#         return result
#
#     def NonMaxSuppression(self, proposal_bbox_inst, confi_thres=0.9,nms_thresh = 0.99, proposal_type="roih"):
#         if proposal_type == "roih":
#             valid_map = proposal_bbox_inst.scores > confi_thres
#
#             # create instances containing boxes and gt_classes
#             image_shape = proposal_bbox_inst.image_size
#             new_proposal_inst = Instances(image_shape)
#
#             # create box  #actually no need valid_map
#             new_bbox_loc = proposal_bbox_inst.pred_boxes.tensor[valid_map, :]
#             new_score = proposal_bbox_inst.scores[valid_map]
#             new_class = proposal_bbox_inst.pred_classes[valid_map]
#
#             # new_bbox_loc = proposal_bbox_inst.pred_boxes.tensor
#             # new_score = proposal_bbox_inst.scores
#             # new_class = proposal_bbox_inst.pred_classes
#             scores,index = new_score.sort(descending = True)
#             keep_inds = []
#             while(len(index) > 0):
#                 cur_inx = index[0]
#                 cur_score = scores[cur_inx]
#                 if cur_score < confi_thres:
#                     index = index[1:]
#                     continue
#                 keep = True
#                 for ind in keep_inds:
#                     current_bbox = new_bbox_loc[cur_inx]
#                     remain_box = new_bbox_loc[ind]
#                     # iou = 1
#                     ioc = self.box_ioc_xyxy(current_bbox,remain_box)
#                     if ioc > nms_thresh:
#                         keep = False
#                         break
#                 if keep:
#                     keep_inds.append(cur_inx)
#                 index = index[1:]
#             # if len(keep_inds) == 0:
#             #     valid_map = proposal_bbox_inst.scores > thres
#             #
#             #     # create instances containing boxes and gt_classes
#             #     image_shape = proposal_bbox_inst.image_size
#             #     new_proposal_inst = Instances(image_shape)
#             #
#             #     # create box
#             #     new_bbox_loc = proposal_bbox_inst.pred_boxes.tensor[valid_map, :]
#             #     new_boxes = Boxes(new_bbox_loc)
#             #
#             #     # add boxes to instances
#             #     new_proposal_inst.gt_boxes = new_boxes
#             #     new_proposal_inst.gt_classes = proposal_bbox_inst.pred_classes[valid_map]
#             #     new_proposal_inst.scores = proposal_bbox_inst.scores[valid_map]
#             #     for i in new_proposal_inst.scores:
#             #         i = 0
#             #     return new_proposal_inst
#
#             keep_inds = torch.tensor(keep_inds)
#             score_nms = new_score[keep_inds.long()]
#             # score_nms = score_nms.reshape(-1,1)
#             # score_nms = score_nms.reshape(-1)
#             box_nms = new_bbox_loc[keep_inds.long()]
#             box_nms = box_nms.reshape(-1,4)
#             box_nms = Boxes(box_nms)
#             class_nms = new_class[keep_inds.long()]
#             # class_nms = class_nms.reshape(-1,1)
#             new_proposal_inst.gt_boxes = box_nms
#             new_proposal_inst.gt_classes = class_nms
#             new_proposal_inst.scores = score_nms
#
#         elif proposal_type == "rpn":
#
#             raise ValueError("Unknown NMS branches")
#
#         return new_proposal_inst
#
#     def process_pseudo_label(
#             self, proposals_rpn_unsup_k, cur_threshold, proposal_type, psedo_label_method=""
#     ):
#         list_instances = []
#         num_proposal_output = 0.0
#         for proposal_bbox_inst in proposals_rpn_unsup_k:
#             # thresholding
#             if psedo_label_method == "thresholding":
#                 proposal_bbox_inst = self.threshold_bbox(
#                     proposal_bbox_inst, thres=cur_threshold, proposal_type=proposal_type
#                 )
#             elif psedo_label_method == "NMS":
#                 proposal_bbox_inst = self.NonMaxSuppression(
#                     proposal_bbox_inst, confi_thres=cur_threshold, proposal_type=proposal_type
#                 )
#
#             else:
#                 raise ValueError("Unkown pseudo label boxes methods")
#             num_proposal_output += len(proposal_bbox_inst)
#             list_instances.append(proposal_bbox_inst)
#         num_proposal_output = num_proposal_output / len(proposals_rpn_unsup_k)
#         return list_instances, num_proposal_output
#
#     def remove_label(self, label_data):
#         for label_datum in label_data:
#             if "instances" in label_datum.keys():
#                 del label_datum["instances"]
#         return label_data
#
#     def add_label(self, unlabled_data, label):
#         for unlabel_datum, lab_inst in zip(unlabled_data, label):
#             unlabel_datum["instances"] = lab_inst
#         return unlabled_data
#
#     def get_label(self, label_data):
#         label_list = []
#         for label_datum in label_data:
#             if "instances" in label_datum.keys():
#                 label_list.append(copy.deepcopy(label_datum["instances"]))
#
#         return label_list
#     # def consistency_compare(self,pesudo_proposals_roih_unsup_k, gt_proposal, cur_compare_threshold):
#     #     consistency = 0.5
#     #
#     #
#     #     return consistency
#
#     # def consistency_compare(self, roi_preds, gt_label, threshold):
#
#
#     # def get_label_test(self, label_data):
#     #     label_list = []
#     #     for label_datum in label_data:
#     #         if "instances" in label_datum.keys():
#     #             label_list.append(label_datum["instances"])
#
#     # =====================================================
#     # =================== Training Flow ===================
#     # =====================================================
#
#
#     # def MixupDetection(self,img1,img2,label1,label2,lambd):
#     #     # mixup two images
#     #     height = max(img1.shape[0], img2.shape[0])
#     #     width = max(img1.shape[1], img2.shape[1])
#     #     mix_img = mx.nd.zeros(shape=(height, width, 3), dtype='float32')
#     #     mix_img[:img1.shape[0], :img1.shape[1], :] = img1.astype('float32') * lambd
#     #     mix_img[:img2.shape[0], :img2.shape[1], :] += img2.astype('float32') * (1. - lambd)
#     #     mix_img = mix_img.astype('uint8')
#     #     y1 = np.hstack((label1, np.full((label1.shape[0], 1), lambd)))
#     #     y2 = np.hstack((label2, np.full((label2.shape[0], 1), 1. - lambd)))
#     #     mix_label = np.vstack((y1, y2))
#     #     return mix_img, mix_label
#
#     def run_step_full_semisup(self,teacher_ds,teacher_dp,):
#         self._trainer.iter = self.iter
#         assert self.model.training, "[UBTeacherTrainer] model was changed to eval mode!"
#         start = time.perf_counter()
#         data = next(self._trainer._data_loader_iter)
#         # data_q and data_k from different augmentations (q:strong, k:weak)
#         # label_strong, label_weak, unlabed_strong, unlabled_weak
#         label_train_data_q, label_train_data_k, label_compare_data_q, label_compare_data_k, unlabel_data_q, unlabel_data_k = data
#         data_time = time.perf_counter() - start
#         name = 520
#         # pretained stage
#         # firstly,copy a teacher net ,update teacher model per iter
#         # train random initialed model(T/S) 4000 iter to get a completely result
#         # compared to teacher ds and teacher dp
#         if self.iter < self.cfg.SEMISUPNET.INITIAL_ITER:
#             # teacher is just working as a model with EMA without teaching
#             if(self.iter >= self.cfg.SEMISUPNET.BURN_UP_STEP):
#                 if self.iter == self.cfg.SEMISUPNET.BURN_UP_STEP:
#                         # update copy the the whole model
#                         self._update_teacher_model(keep_rate=0.00)
#                 #         # self.model.build_discriminator()
#                 #
#                 elif (self.iter - self.cfg.SEMISUPNET.BURN_UP_STEP
#                     ) % self.cfg.SEMISUPNET.TEACHER_UPDATE_ITER == 0:
#                         self._update_teacher_model(
#                             keep_rate=self.cfg.SEMISUPNET.EMA_KEEP_RATE)
#
#             record_dict = {}
#             ######################## For probe #################################
#             # import pdb; pdb. set_trace()
#             gt_unlabel_k = self.get_label(unlabel_data_k)
#             # gt_unlabel_q = self.get_label_test(unlabel_data_q)
#
#             all_label_train_data = label_train_data_q + label_train_data_k
#             # prepare a copy for training
#             all_label_compare_data = label_compare_data_q + label_compare_data_k
#             #  0. remove unlabeled data labels
#             record_all_label_train_data, src_invairent_feature, _, _ = self.model(
#                 all_label_train_data, branch="supervised"
#             )
#             record_dict.update(record_all_label_train_data)
#             # compare_label
#             record_all_label_compare_data, tgt_invarient_feature, _, _ = self.model(
#                 all_label_compare_data, branch="supervised_compare"
#             )
#             new_record_all_label_compare_data = {}
#             for key in record_all_label_compare_data.keys():
#                 new_record_all_label_compare_data[key + "_compare"] = record_all_label_compare_data[
#                     key
#                 ]
#                 record_dict.update(new_record_all_label_compare_data)
#             loss_dict = {}
#             for key in record_dict.keys():
#                 if key[:4] == "loss":
#                     loss_dict[key] = record_dict[key] * 1
#             losses = sum(loss_dict.values())
#             # losses = sum(loss_dict.values())
#
#             metrics_dict = loss_dict
#             metrics_dict["name"] = name
#             metrics_dict["data_time"] = data_time
#
#             self._write_metrics(metrics_dict)
#             # student loss
#             self.optimizer.zero_grad()
#             losses.backward()
#             self.optimizer.step()
#
#         # Dual teacher consistency regularzation with discrimiator updating to keep complimentarity
#         else:
#             if self.iter == self.cfg.SEMISUPNET.INITIAL_ITER:
#                     # update copy the the whole teacher stable model as new student to study
#                 self._update_student_model(keep_rate=0.00)
#             # _extreme_fusion
#             # if (self.iter - self.cfg.SEMISUPNET.INITIAL_ITER
#             # )  == 0:
#             #     # self._update_teacher_model(
#             #     #     keep_rate=self.cfg.SEMISUPNET.EMA_KEEP_RATE)
#             #     self._extreme_fusion(teacher_ds,teacher_dp,keep_rate=0.5)
#
#             elif (self.iter - self.cfg.SEMISUPNET.INITIAL_ITER
#             ) % self.cfg.SEMISUPNET.UPDATE_ITER ==0:
#                 self._update_teacher_model(
#                         keep_rate=self.cfg.SEMISUPNET.EMA_KEEP_RATE)
#                 # self._update_teacher_ds_model(teacher_ds,
#                 #                               keep_rate=self.cfg.SEMISUPNET.EMA_KEEP_RATE)
#                 # self._update_teacher_dp_model(teacher_dp,
#                 #                               keep_rate=self.cfg.SEMISUPNET.EMA_KEEP_RATE)
#
#             record_dict = {}
#
#             ######################## For probe #################################
#             # import pdb; pdb. set_trace()
#             gt_unlabel_k = self.get_label(unlabel_data_k)
#             # gt_unlabel_q = self.get_label_test(unlabel_data_q)
#
#             #  0. remove unlabeled data labels
#             unlabel_data_q = self.remove_label(unlabel_data_q)
#             unlabel_data_k = self.remove_label(unlabel_data_k)
#
#             # with torch.no_grad():
#             #     (
#             #         _,
#             #         proposals_rpn_unsup_k,
#             #         proposals_roih_unsup_k,
#             #         _,
#             #     ) = teacher_ds(unlabel_data_k, branch="unsup_data_weak")
#             #     (
#             #         _,
#             #         proposals_rpn_unsup_k_p,
#             #         proposals_roih_unsup_k_p,
#             #         _,
#             #     ) = teacher_dp(unlabel_data_k, branch="unsup_data_weak")
#             # #
#             # # # todo:pseudo label fusion
#             # pesudo_proposals_roih_unsup_k, _ = self.Knowlegde_Fusion(
#             #     proposals_roih_unsup_k, proposals_roih_unsup_k_p, self.cfg.SEMISUPNET.FUSION_IOU_THR,
#             #     self.cfg.SEMISUPNET.FUSION_BBOX_THRESHOLD, self.cfg.SEMISUPNET.FUSION_WEIGHT
#             # )
#             cur_threshold = self.cfg.SEMISUPNET.BBOX_THRESHOLD
#             with torch.no_grad():
#                 (
#                     _,
#                     proposals_rpn_unsup_k,
#                     proposals_roih_unsup_k,
#                     _,
#                 ) = teacher_ds(unlabel_data_k, branch="unsup_data_weak")
#             pesudo_proposals_roih_unsup_k, _ = self.process_pseudo_label(
#                 proposals_roih_unsup_k, cur_threshold, "roih", "thresholding"
#             )
#             #  2. Pseudo-labeling
#             joint_proposal_dict = {}
#             joint_proposal_dict["proposals_pseudo_roih"] = pesudo_proposals_roih_unsup_k
#             unlabel_data_q = self.add_label(
#                 unlabel_data_q, joint_proposal_dict["proposals_pseudo_roih"]
#             )
#             unlabel_data_k = self.add_label(
#                 unlabel_data_k, joint_proposal_dict["proposals_pseudo_roih"]
#             )
#
#             all_label_data = label_train_data_q + label_train_data_k
#
#             # 4. input both strongly and weakly augmented labeled data into student model
#             record_all_label_data, _, _, _ = self.model(
#                 all_label_data, branch="supervised"
#             )
#             record_dict.update(record_all_label_data)
#
#             all_unlabel_data = unlabel_data_q
#             record_all_unlabel_data, _, _, _ = self.model(
#                 all_unlabel_data, branch="supervised_target"
#             )
#
#             new_record_all_unlabel_data = {}
#             for key in record_all_unlabel_data.keys():
#                 new_record_all_unlabel_data[key + "_pseudo"] = record_all_unlabel_data[
#                     key
#                 ]
#             record_dict.update(new_record_all_unlabel_data)
#
#             for i_index in range(len(unlabel_data_k)):
#                 for k, v in unlabel_data_k[i_index].items():
#                     label_train_data_k[i_index][k + "_unlabeled"] = v
#
#             # all_domain_data = label_train_data_k
#             # record_all_domain_data, _, _, _ = self.model(all_domain_data, branch="domain")
#             # record_dict.update(record_all_domain_data)
#
#             # weight losses
#             loss_dict = {}
#             for key in record_dict.keys():
#                 if key.startswith("loss"):
#                     if key == "loss_rpn_loc_pseudo" or key == "loss_box_reg_pseudo":
#                         # pseudo bbox regression <- 0
#                         loss_dict[key] = record_dict[key] * 0
#                     elif key[-6:] == "pseudo":  # unsupervised loss
#                         loss_dict[key] = (
#                                 record_dict[key] *
#                                 self.cfg.SEMISUPNET.UNSUP_LOSS_WEIGHT
#                         )
#                     elif (
#                             key == "loss_D_img_s" or key == "loss_D_img_t"):
#                         # set weight for discriminator
#                         loss_dict[key] = record_dict[
#                                              key] * self.cfg.SEMISUPNET.DIS_LOSS_WEIGHT  # Need to modify defaults and yaml
#                     else:  # supervised loss
#                         loss_dict[key] = record_dict[key] * self.cfg.SEMISUPNET.SUP_LOSS_WEIGHT
#
#
#             losses = sum(loss_dict.values())
#
#             metrics_dict = loss_dict
#             metrics_dict["name"] = name
#             metrics_dict["data_time"] = data_time
#
#             #todo:student loss
#             self.optimizer.zero_grad()
#             losses.backward()
#             self.optimizer.step()
#
#             # todo:teacher_drd_loss
#             # loss_DRD_dis = teacher_dp(all_domain_data, branch='domain_DRD')
#             # metrics_dict["dis_DRD"] = sum(loss_DRD_dis.values)
#             #
#             # self.optimizer.zero_grad()
#             # loss_DRD = sum(loss_DRD_dis.values()) * self.cfg.SEMISUPNET.DIS_LOSS_WEIGHT * (1-self.cfg.SEMISUPNET.EMA_KEEP_RATE)
#             # loss_DRD.backward()
#             # self.optimizer.step()
#             #
#             # # todo:teacher_did_loss
#             # loss_DID_dis = teacher_ds(all_domain_data, branch='domain_DID')
#             # metrics_dict["dis_DID"] = sum(loss_DID_dis.values)
#             #
#             # self.optimizer.zero_grad()
#             # loss_DID = sum(loss_DID_dis.values()) * self.cfg.SEMISUPNET.DIS_LOSS_WEIGHT * (1-self.cfg.SEMISUPNET.EMA_KEEP_RATE)
#             # loss_DID.backward()
#             # self.optimizer.step()
#
#
#
#
#             self._write_metrics(metrics_dict)
# #         if len(t_ins_list)%800 == 0:
# #             assert len(t_list) == len(t_ins_list) == len(mIOU_list) == len(recall_list) == len(precise_list)
# #
# #             #target img_dis scores for recall/precise
# #             t_dis = []
# #             t_ins_dis = []
# #             bar_recall_t = collections.Counter(t_dis)
# #             bar_precis_t = collections.Counter(t_dis)
# #
# #             bar_recall_t_ins = collections.Counter(t_ins_dis)
# #             bar_precis_t_ins = collections.Counter(t_ins_dis)
# #             # bar_num = len(x_train)
# #             # for i in range(0, 11):
# #             #     bar_recall_t[i] = 0
# #             #     bar_precis_t[i] = 0
# #             for ele in t_list:
# #                 t_dis.append(int(np.round(ele * 10)))
# #
# #             for ele in t_ins_list:
# #                 t_ins_dis.append(int(np.round(ele * 10)))
# #
# #             len_bar = collections.Counter(t_dis)
# #             len_ins_bar = collections.Counter(t_ins_dis)
# #             for i in range(len(t_dis)):
# #                 bar_recall_t[t_dis[i]] += recall_list[i]
# #                 bar_precis_t[t_dis[i]] += precise_list[i]
# #
# #                 bar_recall_t_ins[t_ins_dis[i]] += recall_list[i]
# #                 bar_precis_t_ins[t_ins_dis[i]] += precise_list[i]
# #
# #             pre_t_img_show = []
# #             rec_t_img_show = []
# #
# #             pre_t_ins_show = []
# #             rec_t_ins_show = []
# #             for i in range(0,11):
# #                 s_p = bar_precis_t[i]/(len_bar[i]+0.001)
# #                 s_r = bar_recall_t[i]/(len_bar[i]+0.001)
# #
# #                 s_ins_p = bar_precis_t_ins[i] / (len_ins_bar[i] + 0.001)
# #                 s_ins_r = bar_recall_t_ins[i] / (len_ins_bar[i] + 0.001)
# #
# #                 pre_t_img_show.append(s_p)
# #                 rec_t_img_show.append(s_r)
# #
# #                 pre_t_ins_show.append(s_ins_p)
# #                 rec_t_ins_show.append(s_ins_r)
# #
# #             #source_list_level_each
# #             plt.figure(8)
# #             level_show = dict(p2=[], p3=[], p4=[], p5=[], p6=[])
# #             list_level_source = dict(p2=[], p3=[], p4=[], p5=[], p6=[])
# #             levels = ["p2", "p3", "p4", "p5", "p6"]
# #             for level in levels:
# #                 for ele in source_level_diction_list[level]:
# #                     list_level_source[level].append(int(np.round(ele * 10)))
# #             x_axis = np.arange(11).astype(dtype=np.str)
# #             for level in levels:
# #                 x_train = collections.Counter(list_level_source[level])
# #                 for i in range(0, 11):
# #                     s = x_train[i]
# #                     level_show[level].append(s)
# #             plt.subplot(851)
# #             plt.bar(x_axis, level_show["p2"], width=0.5)
# #             plt.title('source_p2')
# #             plt.subplot(852)
# #             plt.bar(x_axis, level_show["p3"], width=0.5)
# #             plt.title('source_p3')
# #             plt.subplot(853)
# #             plt.bar(x_axis, level_show["p4"], width=0.5)
# #             plt.title('source_p4')
# #             plt.subplot(854)
# #             plt.bar(x_axis, level_show["p5"], width=0.5)
# #             plt.title('source_p5')
# #             plt.subplot(855)
# #             plt.bar(x_axis, level_show["p6"], width=0.5)
# #             plt.title('source_p6')
# #             plt.savefig("source_level_img_bar_iter_{}.png".format(len(t_ins_list) / 4))
# #
# # #target_distribution
# #             plt.figure(9)
# #             level_show_t = dict(p2=[], p3=[], p4=[], p5=[], p6=[])
# #             list_level_target = dict(p2=[], p3=[], p4=[], p5=[], p6=[])
# #             for level in levels:
# #                 for ele in level_diction_list[level]:
# #                     list_level_target[level].append(int(np.round(ele * 10)))
# #             for level in levels:
# #                 x_train = collections.Counter(list_level_target[level])
# #                 for i in range(0, 11):
# #                     s = x_train[i]
# #                     level_show_t[level].append(s)
# #             plt.subplot(951)
# #             plt.bar(x_axis, level_show_t["p2"], width=0.5)
# #             plt.title('source_p2')
# #             plt.subplot(952)
# #             plt.bar(x_axis, level_show_t["p3"], width=0.5)
# #             plt.title('source_p3')
# #             plt.subplot(953)
# #             plt.bar(x_axis, level_show_t["p4"], width=0.5)
# #             plt.title('source_p4')
# #             plt.subplot(954)
# #             plt.bar(x_axis, level_show_t["p5"], width=0.5)
# #             plt.title('source_p5')
# #             plt.subplot(955)
# #             plt.bar(x_axis, level_show_t["p6"], width=0.5)
# #             plt.title('source_p6')
# #             plt.savefig("target_level_img_bar_iter_{}.png".format(len(t_ins_list) / 4))
# #
# #
# #             #img
# #             plt.figure(6)
# #             plt.subplot(621)
# #             plt.bar(x_axis,pre_t_img_show,width=0.5)
# #             plt.xlabel('t_img_dis')
# #             plt.ylabel('precise_score')
# #             plt.title('target_dis_img & precise')
# #
# #             plt.subplot(622)
# #             plt.bar(x_axis, rec_t_img_show, width=0.5)
# #             plt.xlabel('t_img_dis')
# #             plt.ylabel('recall_score')
# #             plt.title('target_dis_img & recall')
# #             plt.savefig("prediction_quailty_with_domain_img_bar_iter_{}.png".format(len(t_ins_list) / 4))
# #
# #             #ins
# #             plt.figure(7)
# #             plt.subplot(721)
# #             plt.bar(x_axis, pre_t_ins_show, width=0.5)
# #             plt.xlabel('t_ins_dis')
# #             plt.ylabel('precise_score')
# #             plt.title('target_dis_ins & precise')
# #
# #             plt.subplot(722)
# #             plt.bar(x_axis, rec_t_ins_show, width=0.5)
# #             plt.xlabel('t_ins_dis')
# #             plt.ylabel('recall_score')
# #             plt.title('target_dis_ins & recall')
# #             plt.savefig("prediction_quailty_with_domain_ins_bar_iter_{}.png".format(len(t_ins_list) / 4))
# #
# #             #scatter for miou and dis_img
# #             plt.figure(1)
# #
# #
# #             plt.subplot(133)
# #             plt.scatter(mIOU_list, t_list,s = 1)
# #             plt.xlabel('mIOU')
# #             plt.ylabel('img_dis')
# #             plt.title('mIOU & domain_img')
# #             plt.savefig("prediction_quailty_with_domain_img_iter_{}.png".format(len(t_ins_list) / 4))
# #             #
# #
# #             #scatter for miou and dis_ins
# #             plt.figure(4)
# #
# #
# #             plt.subplot(433)
# #             plt.scatter(mIOU_list, t_ins_list,s=1)
# #             plt.xlabel('mIOU')
# #             plt.ylabel('ins_dis')
# #             plt.title('iou & domain_ins')
# #             #
# #             plt.savefig("prediction_quailty_with_domain_ins_iter_{}.png".format(len(t_ins_list)/4))
# #
# #
# #             #bar of s_ins_num and s_img_num
# #
# #             plt.figure(5)
# #             plt.subplot(531)
# #             list_ins_source = []
# #             ppt_s_ins = []
# #             for ele in s_ins_list:
# #                 list_ins_source.append(int(np.round(ele * 10)))
# #             x_train = collections.Counter(list_ins_source)
# #             x_num = len(x_train)
# #             x_axis = np.arange(11).astype(dtype=np.str)
# #             for i in range(0, 11):
# #                 s = x_train[i]
# #                 ppt_s_ins.append(s)
# #             plt.bar(x_axis, ppt_s_ins, width=0.5)
# #             # plt.savefig('ins_s.png')
# #             plt.xlabel('s_ins')
# #             plt.ylabel('dis_num')
# #             plt.title('source ins')
# #
# #             plt.subplot(532)
# #             list_img_source = []
# #             ppt_s_img = []
# #             for ele in s_list:
# #                 list_img_source.append(int(np.round(ele * 10)))
# #             x_train = collections.Counter(list_img_source)
# #             x_num = len(x_train)
# #             x_axis = np.arange(11).astype(dtype=np.str)
# #             for i in range(0, 11):
# #                 s = x_train[i]
# #                 ppt_s_img.append(s)
# #             plt.bar(x_axis, ppt_s_img, width=0.5)
# #             # plt.savefig('ins_s.png')
# #             plt.xlabel('s_img')
# #             plt.ylabel('dis_num')
# #             plt.title('source img')
# #
# #             plt.savefig("source_iter_reiou_{}.png".format(len(t_ins_list) / 4))
# #
# #
# #             #bar of target_ins_num and target_img_num
# #             plt.figure(3)
# #
# #             plt.subplot(331)
# #             list_ins_target = []
# #             ppt_t_ins = []
# #             for ele in t_ins_list:
# #                 try:
# #                     list_ins_target.append(int(np.round(ele * 10)))
# #                 except:
# #                     continue
# #             x_train = collections.Counter(list_ins_target)
# #             x_num = len(x_train)
# #             x_axis = np.arange(11).astype(dtype=np.str)
# #             for i in range(0, 11):
# #                 s = x_train[i]
# #                 ppt_t_ins.append(s)
# #             plt.bar(x_axis, ppt_t_ins, width=0.5)
# #             # plt.savefig('ins_s.png')
# #             plt.xlabel('t_ins')
# #             plt.ylabel('dis_num')
# #             plt.title('target ins')
# #
# #             plt.subplot(332)
# #             list_img_target = []
# #             ppt_t_img = []
# #             for ele in t_list:
# #                 list_img_target.append(int(np.round(ele * 10)))
# #             x_train = collections.Counter(list_img_target)
# #             x_num = len(x_train)
# #             x_axis = np.arange(11).astype(dtype=np.str)
# #             for i in range(0, 11):
# #                 s = x_train[i]
# #                 ppt_t_img.append(s)
# #             plt.bar(x_axis, ppt_t_img, width=0.5)
# #             # plt.savefig('ins_s.png')
# #             plt.xlabel('t_img')
# #             plt.ylabel('dis_num')
# #             plt.title('target img')
# #
# #             plt.savefig("target_iter_reiou_{}.png".format(len(t_ins_list) / 4))
#
#
#
#             # plt.subplot(331)
#             # plt.scatter(s_ins_list, s_list, s=1)
#             # plt.xlabel('S_ins')
#             # plt.ylabel('S_img')
#             # plt.title('source ins & img')
#
#             # plt.subplot(332)
#             # plt.scatter(t_ins_list, t_list, s=1)
#             # plt.xlabel('t_ins')
#             # plt.ylabel('t_img')
#             # plt.title('target ins & img')
#
#
#             # plt.savefig("source_iter_{}.png".format(len(t_ins_list) / 4))
#
#         # return psu_sum
#
#         # self._trainer.iter = self.iter
#         # assert self.model.training, "[UBTeacherTrainer] model was changed to eval mode!"
#         # start = time.perf_counter()
#         # data = next(self._trainer._data_loader_iter)
#         # # data_q and data_k from different augmentations (q:strong, k:weak)
#         # # label_strong, label_weak, unlabed_strong, unlabled_weak
#         # label_data_q, label_data_k,label_compare_data_q,label_compare_data_k, unlabel_data_q, unlabel_data_k = data
#         # data_time = time.perf_counter() - start
#         #
#         # # burn-in stage (supervised training with labeled data)
#         # if self.iter < self.cfg.SEMISUPNET.BURN_UP_STEP:
#         #
#         #     # input both strong and weak supervised data into model
#         #     label_data_q.extend(label_data_k)
#         #     record_dict, _, _, _ = self.model(
#         #         label_data_q, branch="supervised")
#         #
#         #     # weight losses
#         #     loss_dict = {}
#         #     for key in record_dict.keys():
#         #         if key[:4] == "loss":
#         #             loss_dict[key] = record_dict[key] * 1
#         #     losses = sum(loss_dict.values())
#         #
#         # else:
#         #     if self.iter == self.cfg.SEMISUPNET.BURN_UP_STEP:
#         #         # update copy the the whole model
#         #         self._update_teacher_model(keep_rate=0.00)
#         #         # self.model.build_discriminator()
#         #
#         #     elif (
#         #         self.iter - self.cfg.SEMISUPNET.BURN_UP_STEP
#         #     ) % self.cfg.SEMISUPNET.TEACHER_UPDATE_ITER == 0:
#         #         self._update_teacher_model(
#         #             keep_rate=self.cfg.SEMISUPNET.EMA_KEEP_RATE)
#         #
#         #     record_dict = {}
#         #
#         #     ######################## For probe #################################
#         #     # import pdb; pdb. set_trace()
#         #     gt_unlabel_k = self.get_label(unlabel_data_k)
#         #     # gt_unlabel_q = self.get_label_test(unlabel_data_q)
#         #
#         #
#         #     #  0. remove unlabeled data labels
#         #     unlabel_data_q = self.remove_label(unlabel_data_q)
#         #     unlabel_data_k = self.remove_label(unlabel_data_k)
#         #
#         #     #  1. generate the pseudo-label using teacher model
#         #     with torch.no_grad():
#         #         (
#         #             _,
#         #             proposals_rpn_unsup_k,
#         #             proposals_roih_unsup_k,
#         #             _,
#         #         ) = self.model_teacher(unlabel_data_k, branch="unsup_data_weak")
#         #
#         #     ######################## For probe #################################
#         #     # import pdb; pdb. set_trace()
#         #
#         #     # probe_metrics = ['compute_fp_gtoutlier', 'compute_num_box']
#         #     # probe_metrics = ['compute_num_box']
#         #     # analysis_pred, _ = self.probe.compute_num_box(gt_unlabel_k,proposals_roih_unsup_k,'pred')
#         #     # record_dict.update(analysis_pred)
#         #     ######################## For probe END #################################
#         #
#         #     #  2. Pseudo-labeling
#         #     cur_threshold = self.cfg.SEMISUPNET.BBOX_THRESHOLD
#         #
#         #     joint_proposal_dict = {}
#         #     joint_proposal_dict["proposals_rpn"] = proposals_rpn_unsup_k
#         #     #Process pseudo labels and thresholding
#         #     (
#         #         pesudo_proposals_rpn_unsup_k,
#         #         nun_pseudo_bbox_rpn,
#         #     ) = self.process_pseudo_label(
#         #         proposals_rpn_unsup_k, cur_threshold, "rpn", "thresholding"
#         #     )
#         #     # analysis_pred, _ = self.probe.compute_num_box(gt_unlabel_k,pesudo_proposals_rpn_unsup_k,'pred',True)
#         #     # record_dict.update(analysis_pred)
#         #
#         #     joint_proposal_dict["proposals_pseudo_rpn"] = pesudo_proposals_rpn_unsup_k
#         #     # Pseudo_labeling for ROI head (bbox location/objectness)
#         #     pesudo_proposals_roih_unsup_k, _ = self.process_pseudo_label(
#         #         proposals_roih_unsup_k, cur_threshold, "roih", "thresholding"
#         #     )
#         #     joint_proposal_dict["proposals_pseudo_roih"] = pesudo_proposals_roih_unsup_k
#         #
#         #     # 3. add pseudo-label to unlabeled data
#         #
#         #     unlabel_data_q = self.add_label(
#         #         unlabel_data_q, joint_proposal_dict["proposals_pseudo_roih"]
#         #     )
#         #     unlabel_data_k = self.add_label(
#         #         unlabel_data_k, joint_proposal_dict["proposals_pseudo_roih"]
#         #     )
#         #
#         #     all_label_data = label_data_q + label_data_k
#         #     all_unlabel_data = unlabel_data_q
#         #
#         #     # 4. input both strongly and weakly augmented labeled data into student model
#         #     record_all_label_data, _, _, _ = self.model(
#         #         all_label_data, branch="supervised"
#         #     )
#         #     record_dict.update(record_all_label_data)
#         #
#         #     # 5. input strongly augmented unlabeled data into model
#         #     record_all_unlabel_data, _, _, _ = self.model(
#         #         all_unlabel_data, branch="supervised_target"
#         #     )
#         #     new_record_all_unlabel_data = {}
#         #     for key in record_all_unlabel_data.keys():
#         #         new_record_all_unlabel_data[key + "_pseudo"] = record_all_unlabel_data[
#         #             key
#         #         ]
#         #     record_dict.update(new_record_all_unlabel_data)
#         #
#         #     # 6. input weakly labeled data (source) and weakly unlabeled data (target) to student model
#         #     # give sign to the target data
#         #
#         #     for i_index in range(len(unlabel_data_k)):
#         #         # unlabel_data_item = {}
#         #         for k, v in unlabel_data_k[i_index].items():
#         #             # label_data_k[i_index][k + "_unlabeled"] = v
#         #             label_data_k[i_index][k + "_unlabeled"] = v
#         #         # unlabel_data_k[i_index] = unlabel_data_item
#         #
#         #     all_domain_data = label_data_k
#         #     # all_domain_data = label_data_k + unlabel_data_k
#         #     record_all_domain_data, _, _, _ = self.model(all_domain_data, branch="domain")
#         #     record_dict.update(record_all_domain_data)
#         #
#         #
#         #     # weight losses
#         #     loss_dict = {}
#         #     for key in record_dict.keys():
#         #         if key.startswith("loss"):
#         #             if key == "loss_rpn_loc_pseudo" or key == "loss_box_reg_pseudo":
#         #                 # pseudo bbox regression <- 0
#         #                 loss_dict[key] = record_dict[key] * 0
#         #             elif key[-6:] == "pseudo":  # unsupervised loss
#         #                 loss_dict[key] = (
#         #                     record_dict[key] *
#         #                     self.cfg.SEMISUPNET.UNSUP_LOSS_WEIGHT
#         #                 )
#         #             elif (
#         #                 key == "loss_D_img_s" or key == "loss_D_img_t"
#         #             ):  # set weight for discriminator
#         #                 # import pdb
#         #                 # pdb.set_trace()
#         #                 loss_dict[key] = record_dict[key] * self.cfg.SEMISUPNET.DIS_LOSS_WEIGHT #Need to modify defaults and yaml
#         #             else:  # supervised loss
#         #                 loss_dict[key] = record_dict[key] * 1
#         #
#         #     losses = sum(loss_dict.values())
#         #
#         # metrics_dict = record_dict
#         # metrics_dict["data_time"] = data_time
#         # self._write_metrics(metrics_dict)
#         #
#         # self.optimizer.zero_grad()
#         # losses.backward()
#         # self.optimizer.step()
#
#     def _write_metrics(self, metrics_dict: dict):
#         metrics_dict = {
#             k: v.detach().cpu().item() if isinstance(v, torch.Tensor) else float(v)
#             for k, v in metrics_dict.items()
#         }
#
#         # gather metrics among all workers for logging
#         # This assumes we do DDP-style training, which is currently the only
#         # supported method in detectron2.
#         all_metrics_dict = comm.gather(metrics_dict)
#         # all_hg_dict = comm.gather(hg_dict)
#
#         if comm.is_main_process():
#             if "data_time" in all_metrics_dict[0]:
#                 # data_time among workers can have high variance. The actual latency
#                 # caused by data_time is the maximum among workers.
#                 data_time = np.max([x.pop("data_time")
#                                     for x in all_metrics_dict])
#                 self.storage.put_scalar("data_time", data_time)
#
#             # average the rest metrics
#             metrics_dict = {
#                 k: np.mean([x[k] for x in all_metrics_dict])
#                 for k in all_metrics_dict[0].keys()
#             }
#
#             # append the list
#             loss_dict = {}
#             for key in metrics_dict.keys():
#                 if key[:4] == "loss":
#                     loss_dict[key] = metrics_dict[key]
#
#             total_losses_reduced = sum(loss for loss in loss_dict.values())
#
#             self.storage.put_scalar("total_loss", total_losses_reduced)
#             if len(metrics_dict) > 1:
#                 self.storage.put_scalars(**metrics_dict)
#     @torch.no_grad()
#     def _update_student_model(self, keep_rate=0.999):
#         if comm.get_world_size() > 1:
#             student_model_dict = {
#                 'module.' + key: value for key, value in self.model_teacher.state_dict().items()
#             }
#             # for key, value in self.model_teacher.state_dict().items():
#             #     print(key)
#         else:
#             student_model_dict = self.model_teacher.state_dict()
#
#         new_teacher_dict = OrderedDict()
#         for key, value in self.model.state_dict().items():
#             # print(key,"woshishabi")
#             if key in student_model_dict.keys():
#                 new_teacher_dict[key] = (
#                         student_model_dict[key] *
#                         (1 - keep_rate) + value * keep_rate
#                 )
#             else:
#                 raise Exception("{} is not found in student model".format(key))
#
#         self.model.load_state_dict(new_teacher_dict)
#     @torch.no_grad()
#     def _update_teacher_model(self, keep_rate=0.999):
#         if comm.get_world_size() > 1:
#             student_model_dict = {
#                 key[7:]: value for key, value in self.model.state_dict().items()
#             }
#         else:
#             student_model_dict = self.model.state_dict()
#
#         new_teacher_dict = OrderedDict()
#         for key, value in self.model_teacher.state_dict().items():
#             if key in student_model_dict.keys():
#                 new_teacher_dict[key] = (
#                         student_model_dict[key] *
#                         (1 - keep_rate) + value * keep_rate
#                 )
#             else:
#                 raise Exception("{} is not found in student model".format(key))
#
#         self.model_teacher.load_state_dict(new_teacher_dict)
#
#     @torch.no_grad()
#     def _extreme_fusion(self, model_ds, model_dp,keep_rate=0.5):
#         if comm.get_world_size() > 1:
#             model_ds_dict = {
#                 key: value for key, value in model_ds.state_dict().items()
#             }
#             model_dp_dict = {
#                 key: value for key, value in model_ds.state_dict().items()
#             }
#         else:
#             model_ds_dict = model_ds.state_dict()
#             model_dp_dict = model_dp.state_dict()
#
#         new_student_dict = OrderedDict()
#         new_teacher_dict = OrderedDict()
#         for key, value in self.model.state_dict().items():
#             if (key[7:] in model_ds_dict.keys()) and (key[7:] in model_dp_dict.keys()):
#                 new_student_dict[key] = (
#                         model_ds_dict[key[7:]] *
#                         (1 - keep_rate) + model_dp_dict[key[7:]] * keep_rate
#                 )
#                 new_teacher_dict[key[7:]] = (
#                         model_ds_dict[key[7:]] *
#                         (1 - keep_rate) + model_dp_dict[key[7:]] * keep_rate
#                 )
#             else:
#                 raise Exception("{} is not found in student model".format(key))
#
#         self.model.load_state_dict(new_student_dict)
#         self.model_teacher.load_state_dict(new_teacher_dict)
#
#     @torch.no_grad()
#     def _update_teacher_ds_model(self, model_ds,keep_rate=0.9996):
#         if comm.get_world_size() > 1:
#             student_model_dict = {
#                 key: value for key, value in self.model_teacher.state_dict().items()
#             }
#             # for key, value in self.model_teacher.state_dict().items():
#             #     print(key)
#         else:
#             student_model_dict = self.model_teacher.state_dict()
#
#         new_teacher_dict = OrderedDict()
#         for key, value in model_ds.state_dict().items():
#             # print(key)
#             if key in student_model_dict.keys():
#                 new_teacher_dict[key] = (
#                         student_model_dict[key] *
#                         (1 - keep_rate) + value * keep_rate
#                 )
#             else:
#                 raise Exception("{} is not found in student model".format(key))
#
#         model_ds.load_state_dict(new_teacher_dict)
#
#     @torch.no_grad()
#     def _update_teacher_dp_model(self,model_dp, keep_rate=0.9996):
#         if comm.get_world_size() > 1:
#             student_model_dict = {
#                 key: value for key, value in self.model_teacher.state_dict().items()
#             }
#         else:
#             student_model_dict = self.model_teacher.state_dict()
#
#         new_teacher_dict = OrderedDict()
#         for key, value in model_dp.state_dict().items():
#             if key in student_model_dict.keys():
#                 new_teacher_dict[key] = (
#                         student_model_dict[key] *
#                         (1 - keep_rate) + value * keep_rate
#                 )
#             else:
#                 raise Exception("{} is not found in student model".format(key))
#
#         model_dp.load_state_dict(new_teacher_dict)
#
#     @torch.no_grad()
#     def _copy_main_model(self):
#         # initialize all parameters
#         if comm.get_world_size() > 1:
#             rename_model_dict = {
#                 key[7:]: value for key, value in self.model.state_dict().items()
#             }
#             self.model_teacher.load_state_dict(rename_model_dict)
#         else:
#             self.model_teacher.load_state_dict(self.model.state_dict())
#
#     @classmethod
#     def build_test_loader(cls, cfg, dataset_name):
#         return build_detection_test_loader(cfg, dataset_name)
#
#     def build_hooks(self):
#         cfg = self.cfg.clone()
#         cfg.defrost()
#         cfg.DATALOADER.NUM_WORKERS = 0  # save some memory and time for PreciseBN
#
#         ret = [
#             hooks.IterationTimer(),
#             hooks.LRScheduler(self.optimizer, self.scheduler),
#             hooks.PreciseBN(
#                 # Run at the same freq as (but before) evaluation.
#                 cfg.TEST.EVAL_PERIOD,
#                 self.model,
#                 # Build a new data loader to not affect training
#                 self.build_train_loader(cfg),
#                 cfg.TEST.PRECISE_BN.NUM_ITER,
#             )
#             if cfg.TEST.PRECISE_BN.ENABLED and get_bn_modules(self.model)
#             else None,
#         ]
#
#         # Do PreciseBN before checkpointer, because it updates the model and need to
#         # be saved by checkpointer.
#         # This is not always the best: if checkpointing has a different frequency,
#         # some checkpoints may have more precise statistics than others.
#         if comm.is_main_process():
#             ret.append(
#                 hooks.PeriodicCheckpointer(
#                     self.checkpointer, cfg.SOLVER.CHECKPOINT_PERIOD
#                 )
#             )
#
#         def test_and_save_results_student():
#             self._last_eval_results_student = self.test(self.cfg, self.model)
#             _last_eval_results_student = {
#                 k + "_student": self._last_eval_results_student[k]
#                 for k in self._last_eval_results_student.keys()
#             }
#             return _last_eval_results_student
#
#         def test_and_save_results_teacher():
#             self._last_eval_results_teacher = self.test(
#                 self.cfg, self.model_teacher)
#             return self._last_eval_results_teacher
#
#         ret.append(hooks.EvalHook(cfg.TEST.EVAL_PERIOD,
#                                   test_and_save_results_student))
#         ret.append(hooks.EvalHook(cfg.TEST.EVAL_PERIOD,
#                                   test_and_save_results_teacher))
#
#         if comm.is_main_process():
#             # run writers in the end, so that evaluation metrics are written
#             ret.append(hooks.PeriodicWriter(self.build_writers(), period=20))
#         return ret
#
#
#
#
#
#
# #
# # import os
# # import time
# # import logging
# # import torch
# # import matplotlib
# # matplotlib.use('Agg')
# # import matplotlib.pyplot as plt
# # import torch.nn as nn
# # import torch.nn.functional as F
# # from torch.nn.parallel import DistributedDataParallel
# # from fvcore.nn.precise_bn import get_bn_modules
# # import numpy as np
# # from collections import OrderedDict
# # # from matplotlib.font_manager import FontProperties
# # import cv2
# # # import numpy as np
# # import detectron2.utils.comm as comm
# # from detectron2.checkpoint import DetectionCheckpointer
# # from detectron2.engine import DefaultTrainer, SimpleTrainer, TrainerBase
# # from detectron2.engine.train_loop import AMPTrainer
# # from detectron2.utils.events import EventStorage
# # from detectron2.evaluation import verify_results, DatasetEvaluators
# # # from detectron2.evaluation import COCOEvaluator, verify_results, DatasetEvaluators
# #
# # from detectron2.data.dataset_mapper import DatasetMapper
# # from detectron2.engine import hooks
# # from detectron2.structures.boxes import Boxes
# # from detectron2.structures.instances import Instances
# # from detectron2.utils.env import TORCH_VERSION
# # from detectron2.data import MetadataCatalog
# #
# # from adapteacher.data.build import (
# #     build_detection_semisup_train_loader,
# #     build_detection_test_loader,
# #     build_detection_semisup_train_loader_two_crops,
# # )
# # from adapteacher.data.dataset_mapper import DatasetMapperTwoCropSeparate
# # from adapteacher.engine.hooks import LossEvalHook
# # # from adapteacher.modeling.meta_arch.ts_ensemble import EnsembleTSModel
# # from adapteacher.checkpoint.detection_checkpoint import DetectionTSCheckpointer
# # from adapteacher.solver.build import build_lr_scheduler
# # from adapteacher.evaluation import PascalVOCDetectionEvaluator, COCOEvaluator
# # from utils.fft import FDA_source_to_target_np
# # from .probe import OpenMatchTrainerProbe
# # import copy
# # import collections
# # from torch.utils.tensorboard import SummaryWriter
# #
# #
# # # Supervised-only Trainer
# # class BaselineTrainer(DefaultTrainer):
# #     def __init__(self, cfg):
# #         """
# #         Args:
# #             cfg (CfgNode):
# #         Use the custom checkpointer, which loads other backbone models
# #         with matching heuristics.
# #         """
# #         cfg = DefaultTrainer.auto_scale_workers(cfg, comm.get_world_size())
# #         model = self.build_model(cfg)
# #         optimizer = self.build_optimizer(cfg, model)
# #         data_loader = self.build_train_loader(cfg)
# #
# #         if comm.get_world_size() > 1:
# #             model = DistributedDataParallel(
# #                 model, device_ids=[comm.get_local_rank()], broadcast_buffers=False
# #             )
# #
# #         TrainerBase.__init__(self)
# #         self._trainer = (AMPTrainer if cfg.SOLVER.AMP.ENABLED else SimpleTrainer)(
# #             model, data_loader, optimizer
# #         )
# #
# #         self.scheduler = self.build_lr_scheduler(cfg, optimizer)
# #         self.checkpointer = DetectionCheckpointer(
# #             model,
# #             cfg.OUTPUT_DIR,
# #             optimizer=optimizer,
# #             scheduler=self.scheduler,
# #         )
# #         self.start_iter = 0
# #         self.max_iter = cfg.SOLVER.MAX_ITER
# #         self.cfg = cfg
# #
# #         self.register_hooks(self.build_hooks())
# #
# #     def resume_or_load(self, resume=True):
# #         """
# #         If `resume==True` and `cfg.OUTPUT_DIR` contains the last checkpoint (defined by
# #         a `last_checkpoint` file), resume from the file. Resuming means loading all
# #         available states (eg. optimizer and scheduler) and update iteration counter
# #         from the checkpoint. ``cfg.MODEL.WEIGHTS`` will not be used.
# #         Otherwise, this is considered as an independent training. The method will load model
# #         weights from the file `cfg.MODEL.WEIGHTS` (but will not load other states) and start
# #         from iteration 0.
# #         Args:
# #             resume (bool): whether to do resume or not
# #         """
# #         checkpoint = self.checkpointer.resume_or_load(
# #             self.cfg.MODEL.WEIGHTS, resume=resume
# #         )
# #         if resume and self.checkpointer.has_checkpoint():
# #             self.start_iter = checkpoint.get("iteration", -1) + 1
# #             # The checkpoint stores the training iteration that just finished, thus we start
# #             # at the next iteration (or iter zero if there's no checkpoint).
# #         if isinstance(self.model, DistributedDataParallel):
# #             # broadcast loaded data/model from the first rank, because other
# #             # machines may not have access to the checkpoint file
# #             if TORCH_VERSION >= (1, 7):
# #                 self.model._sync_params_and_buffers()
# #             self.start_iter = comm.all_gather(self.start_iter)[0]
# #
# #     def train_loop(self, start_iter: int, max_iter: int):
# #         """
# #         Args:
# #             start_iter, max_iter (int): See docs above
# #         """
# #         logger = logging.getLogger(__name__)
# #         logger.info("Starting training from iteration {}".format(start_iter))
# #
# #         self.iter = self.start_iter = start_iter
# #         self.max_iter = max_iter
# #
# #         with EventStorage(start_iter) as self.storage:
# #             try:
# #                 self.before_train()
# #                 for self.iter in range(start_iter, max_iter):
# #                     self.before_step()
# #                     self.run_step()
# #                     self.after_step()
# #             except Exception:
# #                 logger.exception("Exception during training:")
# #                 raise
# #             finally:
# #                 self.after_train()
# #
# #     def run_step(self):
# #         self._trainer.iter = self.iter
# #
# #         assert self.model.training, "[SimpleTrainer] model was changed to eval mode!"
# #         start = time.perf_counter()
# #
# #         data = next(self._trainer._data_loader_iter)
# #         data_time = time.perf_counter() - start
# #
# #         record_dict, _, _, _ = self.model(data, branch="supervised")
# #
# #         num_gt_bbox = 0.0
# #         for element in data:
# #             num_gt_bbox += len(element["instances"])
# #         num_gt_bbox = num_gt_bbox / len(data)
# #         record_dict["bbox_num/gt_bboxes"] = num_gt_bbox
# #
# #         loss_dict = {}
# #         for key in record_dict.keys():
# #             if key[:4] == "loss" and key[-3:] != "val":
# #                 loss_dict[key] = record_dict[key]
# #
# #         losses = sum(loss_dict.values())
# #
# #         metrics_dict = record_dict
# #         metrics_dict["data_time"] = data_time
# #         self._write_metrics(metrics_dict)
# #
# #         self.optimizer.zero_grad()
# #         losses.backward()
# #         self.optimizer.step()
# #
# #     @classmethod
# #     def build_evaluator(cls, cfg, dataset_name, output_folder=None):
# #         if output_folder is None:
# #             output_folder = os.path.join(cfg.OUTPUT_DIR, "inference")
# #         evaluator_list = []
# #         evaluator_type = MetadataCatalog.get(dataset_name).evaluator_type
# #
# #         if evaluator_type == "coco":
# #             evaluator_list.append(COCOEvaluator(
# #                 dataset_name, output_dir=output_folder))
# #         elif evaluator_type == "pascal_voc":
# #             return PascalVOCDetectionEvaluator(dataset_name)
# #         elif evaluator_type == "pascal_voc_water":
# #             return PascalVOCDetectionEvaluator(dataset_name,
# #                                                target_classnames=["bicycle", "bird", "car", "cat", "dog", "person"])
# #         if len(evaluator_list) == 0:
# #             raise NotImplementedError(
# #                 "no Evaluator for the dataset {} with the type {}".format(
# #                     dataset_name, evaluator_type
# #                 )
# #             )
# #         elif len(evaluator_list) == 1:
# #             return evaluator_list[0]
# #
# #         return DatasetEvaluators(evaluator_list)
# #
# #     @classmethod
# #     def build_train_loader(cls, cfg):
# #         return build_detection_semisup_train_loader(cfg, mapper=None)
# #
# #     @classmethod
# #     def build_test_loader(cls, cfg, dataset_name):
# #         """
# #         Returns:
# #             iterable
# #         """
# #         return build_detection_test_loader(cfg, dataset_name)
# #
# #     def build_hooks(self):
# #         """
# #         Build a list of default hooks, including timing, evaluation,
# #         checkpointing, lr scheduling, precise BN, writing events.
# #
# #         Returns:
# #             list[HookBase]:
# #         """
# #         cfg = self.cfg.clone()
# #         cfg.defrost()
# #         cfg.DATALOADER.NUM_WORKERS = 0
# #
# #         ret = [
# #             hooks.IterationTimer(),
# #             hooks.LRScheduler(self.optimizer, self.scheduler),
# #             hooks.PreciseBN(
# #                 cfg.TEST.EVAL_PERIOD,
# #                 self.model,
# #                 self.build_train_loader(cfg),
# #                 cfg.TEST.PRECISE_BN.NUM_ITER,
# #             )
# #             if cfg.TEST.PRECISE_BN.ENABLED and get_bn_modules(self.model)
# #             else None,
# #         ]
# #
# #         if comm.is_main_process():
# #             ret.append(
# #                 hooks.PeriodicCheckpointer(
# #                     self.checkpointer, cfg.SOLVER.CHECKPOINT_PERIOD
# #                 )
# #             )
# #
# #         def test_and_save_results():
# #             self._last_eval_results = self.test(self.cfg, self.model)
# #             return self._last_eval_results
# #
# #         ret.append(hooks.EvalHook(cfg.TEST.EVAL_PERIOD, test_and_save_results))
# #
# #         if comm.is_main_process():
# #             ret.append(hooks.PeriodicWriter(self.build_writers(), period=20))
# #         return ret
# #
# #     def _write_metrics(self, metrics_dict: dict):
# #         """
# #         Args:
# #             metrics_dict (dict): dict of scalar metrics
# #         """
# #         metrics_dict = {
# #             k: v.detach().cpu().item() if isinstance(v, torch.Tensor) else float(v)
# #             for k, v in metrics_dict.items()
# #         }
# #         # gather metrics among all workers for logging
# #         # This assumes we do DDP-style training, which is currently the only
# #         # supported method in detectron2.
# #         all_metrics_dict = comm.gather(metrics_dict)
# #
# #         if comm.is_main_process():
# #             if "data_time" in all_metrics_dict[0]:
# #                 data_time = np.max([x.pop("data_time")
# #                                     for x in all_metrics_dict])
# #                 self.storage.put_scalar("data_time", data_time)
# #
# #             metrics_dict = {
# #                 k: np.mean([x[k] for x in all_metrics_dict])
# #                 for k in all_metrics_dict[0].keys()
# #             }
# #
# #             loss_dict = {}
# #             for key in metrics_dict.keys():
# #                 if key[:4] == "loss":
# #                     loss_dict[key] = metrics_dict[key]
# #
# #             total_losses_reduced = sum(loss for loss in loss_dict.values())
# #
# #             self.storage.put_scalar("total_loss", total_losses_reduced)
# #             if len(metrics_dict) > 1:
# #                 self.storage.put_scalars(**metrics_dict)
# #
# # def bb_intersection_over_union(A, B):
# #     xA = max(A[0], B[0])
# #     yA = max(A[1], B[1])
# #     xB = min(A[2], B[2])
# #     yB = min(A[3], B[3])
# #
# #     # compute the area of intersection rectangle
# #     interArea = max(0, xB - xA) * max(0, yB - yA)
# #
# #     if interArea == 0:
# #         return 0.0
# #
# #     # compute the area of both the prediction and ground-truth rectangles
# #     boxAArea = (A[2] - A[0]) * (A[3] - A[1])
# #     boxBArea = (B[2] - B[0]) * (B[3] - B[1])
# #
# #     iou = interArea / float(boxAArea + boxBArea - interArea)
# #     return iou
# #
# #
# # def prefilter_boxes(boxes, scores, labels, weights, thr):
# #     # Create dict with boxes stored by its label
# #     new_boxes = dict()
# #     for t in range(len(boxes)):
# #         for j in range(len(boxes[t])):
# #             score = scores[t][j]
# #             if score < thr:
# #                 continue
# #             label = int(labels[t][j])
# #             box_part = boxes[t][j]
# #             # box_area = (box_part[3]-box_part[1]) * (box_part[2]-box_part[0])
# #             b = [int(label), float(score) * weights[t], float(box_part[0]), float(box_part[1]), float(box_part[2]),
# #                  float(box_part[3])]
# #             if label not in new_boxes:
# #                 new_boxes[label] = []
# #             new_boxes[label].append(b)
# #
# #     # Sort each list in dict by score and transform it to numpy array
# #     for k in new_boxes:
# #         current_boxes = np.array(new_boxes[k])
# #         new_boxes[k] = current_boxes[current_boxes[:, 1].argsort()[::-1]]
# #
# #     return new_boxes
# #
# # def get_mIOU(pred_ins, gt_ins, iou_threh = 0.5):
# #     pred_boxs, gt_boxs = pred_ins.gt_boxes.tensor, gt_ins.gt_boxes.tensor
# #     iou = np.zeros((len(pred_boxs), len(gt_boxs)))
# #     #calculate iou for filter
# #     for j in range(len(gt_boxs)):
# #         for i in range(len(pred_boxs)):
# #             iou[i][j] = bb_intersection_over_union(pred_boxs[i],gt_boxs[j])
# #     #filter
# #     iou_list = []
# #     gt_index = iou.argmax(axis=1)
# #     if len(pred_boxs)>0:
# #         pred_index = iou.argmax(axis=0)
# #         for j in range(len(gt_boxs)):
# #             iou_list.append(iou[pred_index][j])
# #     gt_index[iou.max(axis=1) < iou_threh] = -1
# #     select = np.zeros(gt_boxs.size(0), dtype=bool)
# #     match = []
# #     confidence = []
# #     for sid, gt_idx in enumerate(gt_index):
# #         if gt_idx >= 0:
# #             if not select[gt_idx]:
# #                 match.append(1)
# #             else:
# #                 match.append(0)
# #             select[gt_idx] = True
# #         else:
# #             match.append(0)
# #         # iou_list.append(iou[sid][gt_idx])
# #         confidence.append(pred_ins.scores[sid].detach().cpu())
# #
# #     match = np.asarray(match)
# #     confidence = np.asarray(confidence)
# #     iou_list = np.asarray(iou_list)
# #     order = confidence.argsort()[::-1]
# #     match = match[order]
# #
# #     # tp = np.cumsum(match == 1).sum()
# #     # fp = np.cumsum(match == 0).sum()
# #     tp = (match == 1).sum()
# #     fp = (match == 0).sum()
# #
# #     if len(match) != 0 :
# #
# #         rec = tp / (fp + tp)
# #         prec = tp / len(match)
# #         mIOU = iou_list.mean()
# #     else:
# #         rec = 0.0
# #         prec = 0.0
# #         mIOU = 0.0
# # #这个iou是以gt为基准，原本的是按照pred_box，如果是按面积计，他的无疑更合理
# #     return rec, prec, mIOU
# #
# # def get_mIOU2(pred_ins, gt_ins, iou_threh = 0.5):
# #     pred_boxs, gt_boxs = pred_ins.gt_boxes.tensor, gt_ins.gt_boxes.tensor
# #     iou = np.zeros((len(pred_boxs), len(gt_boxs)))
# #     #calculate iou for filter
# #     for j in range(len(gt_boxs)):
# #         for i in range(len(pred_boxs)):
# #             iou[i][j] = bb_intersection_over_union(pred_boxs[i],gt_boxs[j])
# #     #filter
# #     iou_list = []
# #     gt_index = iou.argmax(axis=1)
# #     if len(pred_boxs)>0:
# #         pred_index = iou.argmax(axis=0)
# #         # for j in range(len(gt_boxs)):
# #         #     iou_list.append(iou[pred_index][j])
# #     gt_index[iou.max(axis=1) < iou_threh] = -1
# #     select = np.zeros(gt_boxs.size(0), dtype=bool)
# #     match = []
# #     confidence = []
# #     for sid, gt_idx in enumerate(gt_index):
# #         if gt_idx >= 0:
# #             if not select[gt_idx]:
# #                 match.append(1)
# #                 iou_list.append(iou[sid][gt_idx])
# #             else:
# #                 match.append(0)
# #             select[gt_idx] = True
# #         else:
# #             match.append(0)
# #
# #         confidence.append(pred_ins.scores[sid].detach().cpu())
# #
# #     match = np.asarray(match)
# #     confidence = np.asarray(confidence)
# #     iou_list = np.asarray(iou_list)
# #     order = confidence.argsort()[::-1]
# #     match = match[order]
# #
# #     # tp = np.cumsum(match == 1).sum()
# #     # fp = np.cumsum(match == 0).sum()
# #     tp = (match == 1).sum()
# #     fp = (match == 0).sum()
# #
# #     rec = tp / (fp + tp)
# #     prec = tp / len(match)
# #     mIOU = iou_list.mean()
# # #这个是按面积计miou
# #     return rec, prec, mIOU
# #
# # def get_weighted_box(boxes, conf_type='avg'):
# #     """
# #     Create weighted box for set of boxes
# #     :param boxes: set of boxes to fuse
# #     :param conf_type: type of confidence one of 'avg' or 'max'
# #     :return: weighted box
# #     """
# #     #oral
# #     # box = np.zeros(6, dtype=np.float32)
# #     # conf = 0
# #     # conf_list = []
# #     # for b in boxes:
# #     #     box[2:] += (b[1] * b[2:])
# #     #     conf += b[1]
# #     #     conf_list.append(b[1])
# #     # box[0] = boxes[0][0]
# #     # if conf_type == 'avg':
# #     #     box[1] = conf / len(boxes)
# #     # elif conf_type == 'max':
# #     #     box[1] = np.array(conf_list).max()
# #     # box[2:] /= conf
# #
# #     #area_weights
# #     box = np.zeros(6, dtype=np.float32)
# #     conf = 0
# #     area = 0
# #     i = 0
# #     box_area1 = int((boxes[0][5] - boxes[0][3]) * (boxes[0][4] - boxes[0][2]))
# #     box_area2 = int((boxes[1][5] - boxes[1][3]) * (boxes[1][4] - boxes[1][2]))
# #     # area_weights = [box_area1 / box_area2,1]
# #     conf_list = []
# #     for b in boxes:
# #         # box_area = (b[5] -b[3])* (b[4]-b[2])
# #         box[2:] += (b[1] * b[2:])
# #         conf += b[1]
# #         i+=1
# #         conf_list.append(b[1])
# #     box[0] = boxes[0][0]
# #     if conf_type == 'avg':
# #         box[1] = conf / len(boxes)
# #     elif conf_type == 'max':
# #         box[1] = np.array(conf_list).max()
# #     box[2:] /= (conf )
# #     return box
# #
# #
# # grads = {}
# #
# #
# # # 这个函数是为了获取中间变量的梯度，我方案中的Z不是一个叶子结点，所以其梯度在反向传播之后不会被保存
# # def save_grad(name):
# #     def hook(grad):
# #         grads[name] = grad
# #
# #     return hook
# #
# #
# # def find_matching_box(boxes_list, new_box, match_iou):
# #     best_iou = match_iou
# #     best_index = -1
# #     for i in range(len(boxes_list)):
# #         box = boxes_list[i]
# #         if box[0] != new_box[0]:
# #             continue
# #         iou = bb_intersection_over_union(box[2:], new_box[2:])
# #         if iou > best_iou:
# #             best_index = i
# #             best_iou = iou
# #
# #     return best_index, best_iou
# #
# #
# # def weighted_boxes_fusion(boxes_list, scores_list, labels_list, weights=None, iou_thr=0.55, skip_box_thr=0.8,
# #                           conf_type='avg', allows_overflow=False):
# #     '''
# #     :param boxes_list: list of boxes predictions from each model, each box is 4 numbers.
# #     It has 3 dimensions (models_number, model_preds, 4)
# #     Order of boxes: x1, y1, x2, y2. We expect float normalized coordinates [0; 1]
# #     :param scores_list: list of scores for each model
# #     :param labels_list: list of labels for each model
# #     :param weights: list of weights for each model. Default: None, which means weight == 1 for each model
# #     :param iou_thr: IoU value for boxes to be a match
# #     :param skip_box_thr: exclude boxes with score lower than this variable
# #     :param conf_type: how to calculate confidence in weighted boxes. 'avg': average value, 'max': maximum value
# #     :param allows_overflow: false if we want confidence score not exceed 1.0
# #
# #     :return: boxes: boxes coordinates (Order of boxes: x1, y1, x2, y2).
# #     :return: scores: confidence scores
# #     :return: labels: boxes labels
# #     '''
# #
# #     if weights is None:
# #         weights = np.ones(len(boxes_list))
# #     if len(weights) != len(boxes_list):
# #         print('Warning: incorrect number of weights {}. Must be: {}. Set weights equal to 1.'.format(len(weights),
# #                                                                                                      len(boxes_list)))
# #         weights = np.ones(len(boxes_list))
# #     weights = np.array(weights)
# #
# #     if conf_type not in ['avg', 'max']:
# #         print('Unknown conf_type: {}. Must be "avg" or "max"'.format(conf_type))
# #         exit()
# #         #filter boxes which score>thr
# #     filtered_boxes = prefilter_boxes(boxes_list, scores_list, labels_list, weights, skip_box_thr)
# #     if len(filtered_boxes) == 0:
# #         return np.zeros((0, 4)), np.zeros((0,)), np.zeros((0,))
# #
# #     overall_boxes = []
# #     for label in filtered_boxes:
# #         boxes = filtered_boxes[label]
# #         new_boxes = []
# #         weighted_boxes = []
# #
# #         # Clusterize boxes
# #         for j in range(0, len(boxes)):
# #             index, best_iou = find_matching_box(weighted_boxes, boxes[j], iou_thr)
# #             if index != -1:
# #                 new_boxes[index].append(boxes[j])
# #                 weighted_boxes[index] = get_weighted_box(new_boxes[index], conf_type)
# #             else:
# #                 new_boxes.append([boxes[j].copy()])
# #                 weighted_boxes.append(boxes[j].copy())
# #
# #         # Rescale confidence based on number of models and boxes
# #         for i in range(len(new_boxes)):
# #             if not allows_overflow:
# #                 weighted_boxes[i][1] = weighted_boxes[i][1] * min(weights.sum(), len(new_boxes[i])) / weights.sum()
# #             else:
# #                 weighted_boxes[i][1] = weighted_boxes[i][1] * len(new_boxes[i]) / weights.sum()
# #         overall_boxes.append(np.array(weighted_boxes))
# #
# #     overall_boxes = np.concatenate(overall_boxes, axis=0)
# #     overall_boxes = overall_boxes[overall_boxes[:, 1].argsort()[::-1]]
# #     boxes = overall_boxes[:, 2:]
# #     scores = overall_boxes[:, 1]
# #     labels = overall_boxes[:, 0]
# #     return boxes, scores, labels
# #
# # # Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved
# # from torch.nn.parallel import DataParallel, DistributedDataParallel
# # import torch.nn as nn
# #
# #
# # class EnsembleTSModel(nn.Module):
# #     def __init__(self, modelTeacher, modelStudent):
# #         super(EnsembleTSModel, self).__init__()
# #
# #         if isinstance(modelTeacher, (DistributedDataParallel, DataParallel)):
# #             modelTeacher_ds = modelTeacher.module
# #         # if isinstance(modelTeacher_dp, (DistributedDataParallel, DataParallel)):
# #         #     modelTeacher_dp = modelTeacher_dp.module
# #         if isinstance(modelStudent, (DistributedDataParallel, DataParallel)):
# #             modelStudent = modelStudent.module
# #         # if isinstance(modelDomain, (DistributedDataParallel, DataParallel)):
# #         #     modelStudent = modelDomain.module
# #
# #         self.modelTeacher = modelTeacher
# #         # self.modelTeacher_dp = modelTeacher_dp
# #         self.modelStudent = modelStudent
# #         # self.modelDomain = modelDomain
# #
# # # Adaptive Teacher Trainer
# # class ATeacherTrainer(DefaultTrainer):
# #     def __init__(self, cfg):
# #         """
# #         Args:
# #             cfg (CfgNode):
# #         Use the custom checkpointer, which loads other backbone models
# #         with matching heuristics.
# #         """
# #         cfg = DefaultTrainer.auto_scale_workers(cfg, comm.get_world_size())
# #         data_loader = self.build_train_loader(cfg)
# #
# #         # create an student model
# #         model = self.build_model(cfg)
# #         optimizer = self.build_optimizer(cfg, model)
# #
# #         # create an teacher model
# #         model_teacher = self.build_model(cfg)
# #         self.model_teacher = model_teacher
# #
# #         # model_teacher = self.build_model(cfg)
# #         # self.model_teacher_dp = model_teacher
# #
# #
# #
# #         # model_domain = self.build_model(cfg)
# #         # self.model_domain = model_domain
# #
# #         # For training, wrap with DDP. But don't need this for inference.
# #         if comm.get_world_size() > 1:
# #             model = DistributedDataParallel(
# #                 model, device_ids=[comm.get_local_rank()], broadcast_buffers=False
# #             )
# #
# #         TrainerBase.__init__(self)
# #         self._trainer = (AMPTrainer if cfg.SOLVER.AMP.ENABLED else SimpleTrainer)(
# #             model, data_loader, optimizer
# #         )
# #         self.scheduler = self.build_lr_scheduler(cfg, optimizer)
# #
# #         # Ensemble teacher and student model is for model saving and loading
# #         ensem_ts_model = EnsembleTSModel(model_teacher, model)
# #
# #         self.checkpointer = DetectionTSCheckpointer(
# #             ensem_ts_model,
# #             cfg.OUTPUT_DIR,
# #             optimizer=optimizer,
# #             scheduler=self.scheduler,
# #         )
# #         self.start_iter = 0
# #         self.max_iter = cfg.SOLVER.MAX_ITER
# #         self.cfg = cfg
# #
# #         self.probe = OpenMatchTrainerProbe(cfg)
# #         self.register_hooks(self.build_hooks())
# #
# #     def resume_or_load(self, resume=True):
# #         """
# #         If `resume==True` and `cfg.OUTPUT_DIR` contains the last checkpoint (defined by
# #         a `last_checkpoint` file), resume from the file. Resuming means loading all
# #         available states (eg. optimizer and scheduler) and update iteration counter
# #         from the checkpoint. ``cfg.MODEL.WEIGHTS`` will not be used.
# #         Otherwise, this is considered as an independent training. The method will load model
# #         weights from the file `cfg.MODEL.WEIGHTS` (but will not load other states) and start
# #         from iteration 0.
# #         Args:
# #             resume (bool): whether to do resume or not
# #         """
# #         checkpoint = self.checkpointer.resume_or_load(
# #             self.cfg.MODEL.WEIGHTS, resume=resume
# #         )
# #         # checkpoint_dp = self.checkpointer.resume_or_load(
# #         #     self.cfg.MODEL.WEIGHTS_DP, resume=resume
# #         # )
# #         if resume and self.checkpointer.has_checkpoint():
# #             self.start_iter = checkpoint.get("iteration", -1) + 1
# #             # self.start_iter = checkpoint_dp.get("iteration", -1) + 1
# #             # The checkpoint stores the training iteration that just finished, thus we start
# #             # at the next iteration (or iter zero if there's no checkpoint).
# #         if isinstance(self.model, DistributedDataParallel):
# #             # broadcast loaded data/model from the first rank, because other
# #             # machines may not have access to the checkpoint file
# #             # if TORCH_VERSION >= (1, 7):
# #             #     self.model._sync_params_and_buffers()
# #             self.start_iter = comm.all_gather(self.start_iter)[0]
# #
# #     @classmethod
# #     def build_evaluator(cls, cfg, dataset_name, output_folder=None):
# #         if output_folder is None:
# #             output_folder = os.path.join(cfg.OUTPUT_DIR, "inference")
# #         evaluator_list = []
# #         evaluator_type = MetadataCatalog.get(dataset_name).evaluator_type
# #
# #         if evaluator_type == "coco":
# #             evaluator_list.append(COCOEvaluator(
# #                 dataset_name, output_dir=output_folder))
# #         elif evaluator_type == "pascal_voc":
# #             return PascalVOCDetectionEvaluator(dataset_name)
# #         elif evaluator_type == "pascal_voc_water":
# #             return PascalVOCDetectionEvaluator(dataset_name,
# #                                                target_classnames=["bicycle", "bird", "car", "cat", "dog", "person"])
# #         if len(evaluator_list) == 0:
# #             raise NotImplementedError(
# #                 "no Evaluator for the dataset {} with the type {}".format(
# #                     dataset_name, evaluator_type
# #                 )
# #             )
# #         elif len(evaluator_list) == 1:
# #             return evaluator_list[0]
# #
# #         return DatasetEvaluators(evaluator_list)
# #
# #     @classmethod
# #     def build_train_loader(cls, cfg):
# #         mapper = DatasetMapperTwoCropSeparate(cfg, True)
# #         return build_detection_semisup_train_loader_two_crops(cfg, mapper)
# #
# #     @classmethod
# #     def build_lr_scheduler(cls, cfg, optimizer):
# #         return build_lr_scheduler(cfg, optimizer)
# #
# #     def train(self,teacher_ds,teacher_dp):
# #         self.train_loop(self.start_iter, self.max_iter,teacher_ds,teacher_dp)
# #         if hasattr(self, "_last_eval_results") and comm.is_main_process():
# #             verify_results(self.cfg, self._last_eval_results)
# #             return self._last_eval_results
# #
# #     def train_loop(self, start_iter: int, max_iter: int,teachers_ds,teacher_dp):
# #         logger = logging.getLogger(__name__)
# #         logger.info("Starting training from iteration {}".format(start_iter))
# #
# #         self.iter = self.start_iter = start_iter
# #         self.max_iter = max_iter
# #
# #         with EventStorage(start_iter) as self.storage:
# #             try:
# #                 self.before_train()
# #                 # recall_list = []
# #                 # precise_list = []
# #                 # mIOU_list = []
# #                 # t_ins_score_list = []
# #                 # t_score_list = []
# #                 # s_ins_score_list = []
# #                 # s_score_list = []
# #                 # level_diction_list = dict(p2=[], p3=[], p4=[], p5=[], p6=[])
# #                 # source_level_diction_list = dict(p2=[], p3=[], p4=[], p5=[], p6=[])
# #                 for self.iter in range(start_iter, max_iter):
# #                 # for self.iter in range(2000, max_iter):
# #                     self.before_step()
# #                     self.run_step_full_semisup(teachers_ds,teacher_dp)
# #                     self.after_step()
# #             except Exception:
# #                 logger.exception("Exception during training:")
# #                 raise
# #             finally:
# #                 self.after_train()
# #
# #     def NonMaxSuppression(self, proposal_bbox_inst, confi_thres=0.7,nms_thresh = 0.45, proposal_type="roih"):
# #         if proposal_type == "roih":
# #             valid_map = proposal_bbox_inst.scores > confi_thres
# #
# #             # create instances containing boxes and gt_classes
# #             image_shape = proposal_bbox_inst.image_size
# #             new_proposal_inst = Instances(image_shape)
# #
# #             # create box  #actually no need valid_map
# #             new_bbox_loc = proposal_bbox_inst.pred_boxes.tensor[valid_map, :]
# #             new_score = proposal_bbox_inst.scores[valid_map,:]
# #             new_class = proposal_bbox_inst.pred_classes[valid_map,:]
# #             # new_bbox_loc = proposal_bbox_inst.pred_boxes.tensor
# #             # new_score = proposal_bbox_inst.scores
# #             # new_class = proposal_bbox_inst.pred_classes
# #             scores,index = new_score.sort(descending = True)
# #             keep_inds = []
# #             while(len(index) > 0):
# #                 cur_inx = index[0]
# #                 cur_score = scores[cur_inx]
# #                 if cur_score < confi_thres:
# #                     break;
# #                 keep = True
# #                 for ind in keep_inds:
# #                     current_bbox = new_bbox_loc[cur_inx]
# #                     remain_box = new_bbox_loc[ind]
# #                     # iou = 1
# #                     ioc = self.box_ioc_xyxy(current_bbox,remain_box)
# #                     if ioc > nms_thresh:
# #                         keep = False
# #                         break
# #
# #                 if keep:
# #                     keep_inds.append(cur_inx)
# #                 index = index[1:]
# #             # if len(keep_inds) == 0:
# #             #     valid_map = proposal_bbox_inst.scores > thres
# #             #
# #             #     # create instances containing boxes and gt_classes
# #             #     image_shape = proposal_bbox_inst.image_size
# #             #     new_proposal_inst = Instances(image_shape)
# #             #
# #             #     # create box
# #             #     new_bbox_loc = proposal_bbox_inst.pred_boxes.tensor[valid_map, :]
# #             #     new_boxes = Boxes(new_bbox_loc)
# #             #
# #             #     # add boxes to instances
# #             #     new_proposal_inst.gt_boxes = new_boxes
# #             #     new_proposal_inst.gt_classes = proposal_bbox_inst.pred_classes[valid_map]
# #             #     new_proposal_inst.scores = proposal_bbox_inst.scores[valid_map]
# #             #     for i in new_proposal_inst.scores:
# #             #         i = 0
# #             #     return new_proposal_inst
# #
# #
# #
# #             keep_inds = torch.tensor(keep_inds)
# #             score_nms = new_score[keep_inds.long()]
# #             # score_nms = score_nms.reshape(-1,1)
# #             # score_nms = score_nms.reshape(-1)
# #             box_nms = new_bbox_loc[keep_inds.long()]
# #             box_nms = box_nms.reshape(-1,4)
# #             box_nms = Boxes(box_nms)
# #             class_nms = new_class[keep_inds.long()]
# #             # class_nms = class_nms.reshape(-1,1)
# #             new_proposal_inst.gt_boxes = box_nms
# #             new_proposal_inst.gt_classes = class_nms
# #             new_proposal_inst.scores = score_nms
# #
# #
# #         elif proposal_type == "rpn":
# #
# #             raise ValueError("Unknown NMS branches")
# #
# #         return new_proposal_inst
# #     # =====================================================
# #     # ================== Pseduo-labeling ==================
# #     # =====================================================
# #     def threshold_bbox(self, proposal_bbox_inst, thres=0.7, proposal_type="roih"):
# #         if proposal_type == "rpn":
# #             valid_map = proposal_bbox_inst.objectness_logits > thres
# #
# #             # create instances containing boxes and gt_classes
# #             image_shape = proposal_bbox_inst.image_size
# #             new_proposal_inst = Instances(image_shape)
# #
# #             # create box
# #             new_bbox_loc = proposal_bbox_inst.proposal_boxes.tensor[valid_map, :]
# #             new_boxes = Boxes(new_bbox_loc)
# #
# #             # add boxes to instances
# #             new_proposal_inst.gt_boxes = new_boxes
# #             new_proposal_inst.objectness_logits = proposal_bbox_inst.objectness_logits[
# #                 valid_map
# #             ]
# #         elif proposal_type == "roih":
# #             valid_map = proposal_bbox_inst.scores > thres
# #
# #             # create instances containing boxes and gt_classes
# #             image_shape = proposal_bbox_inst.image_size
# #             new_proposal_inst = Instances(image_shape)
# #
# #             # create box
# #             new_bbox_loc = proposal_bbox_inst.pred_boxes.tensor[valid_map, :]
# #             new_boxes = Boxes(new_bbox_loc)
# #
# #             # add boxes to instances
# #             new_proposal_inst.gt_boxes = new_boxes
# #             new_proposal_inst.gt_classes = proposal_bbox_inst.pred_classes[valid_map]
# #             new_proposal_inst.scores = proposal_bbox_inst.scores[valid_map]
# #
# #         return new_proposal_inst
# #
# #     def box_ioc_xyxy(self,box1_rank,box2_sus):
# #
# #         xA = max(box1_rank[0],box2_sus[0])
# #         yA = max(box1_rank[1],box2_sus[1])
# #         xB = min(box1_rank[2], box2_sus[2])
# #         yB = min(box1_rank[3], box2_sus[3])
# #
# #         intersect = max(0,xB - xA + 1) * max(0,yB - yA + 1)
# #         # box_area1 = (box1[2]-box1[0] + 1) * (box1[3] - box1[1] + 1)
# #         box_area2 = (box2_sus[2] - box2_sus[0] + 1) * (box2_sus[3] - box2_sus[1] + 1)
# #
# #         # ioc = intersect / float(box_area2 + box_area1 -intersect)
# #         ioc = intersect / float(box_area2)
# #         return ioc
# #
# #     def Knowlegde_Fusion(self,proposals_T, proposals_S, iou_thr=0.5, skip_box_thr=0.05, weights=[1, 1]):
# #         assert len(proposals_T) == len(proposals_S)
# #         list_instances = []
# #         num_proposal_output = 0.0
# #         for i in range(len(proposals_T)):
# #             pseudo_label_inst = self.pseudo_fusion(proposals_T[i], proposals_S[i], iou_thr, skip_box_thr, weights)
# #
# #             num_proposal_output += len(pseudo_label_inst)
# #             list_instances.append(pseudo_label_inst)
# #         num_proposal_output = num_proposal_output / (len(proposals_T) + len(proposals_S))
# #         return list_instances, num_proposal_output
# #
# #     def pseudo_fusion(self,output_t, output_s, iou_thr=0.5, skip_box_thr=0.05, weights=[1, 1]):
# #
# #         image_size = output_t.image_size
# #
# #         boxes_list, scores_list, labels_list = [], [], []
# #
# #         box_list_t = output_t.pred_boxes.tensor
# #         scores_list_t = output_t.scores
# #         classes_list_t = output_t.pred_classes
# #
# #         box_list_s = output_s.pred_boxes.tensor
# #         scores_list_s = output_s.scores
# #         classes_list_s = output_s.pred_classes
# #
# #         boxes_list.append(box_list_t)
# #         boxes_list.append(box_list_s)
# #         scores_list.append(scores_list_t)
# #         scores_list.append(scores_list_s)
# #         labels_list.append(classes_list_t)
# #         labels_list.append(classes_list_s)
# #         boxes, scores, labels = weighted_boxes_fusion(boxes_list, scores_list, labels_list, weights=weights,
# #                                                       iou_thr=iou_thr, skip_box_thr=skip_box_thr)
# #         result = Instances(image_size)
# #         boxes = Boxes(torch.tensor(boxes).cuda())
# #         boxes.clip(image_size)
# #         result.gt_boxes = boxes
# #         result.scores = torch.tensor(scores).cuda()
# #         result.gt_classes = torch.tensor(labels).cuda().long()
# #         return result
# #
# #     def NonMaxSuppression(self, proposal_bbox_inst, confi_thres=0.9,nms_thresh = 0.99, proposal_type="roih"):
# #         if proposal_type == "roih":
# #             valid_map = proposal_bbox_inst.scores > confi_thres
# #
# #             # create instances containing boxes and gt_classes
# #             image_shape = proposal_bbox_inst.image_size
# #             new_proposal_inst = Instances(image_shape)
# #
# #             # create box  #actually no need valid_map
# #             new_bbox_loc = proposal_bbox_inst.pred_boxes.tensor[valid_map, :]
# #             new_score = proposal_bbox_inst.scores[valid_map]
# #             new_class = proposal_bbox_inst.pred_classes[valid_map]
# #
# #             # new_bbox_loc = proposal_bbox_inst.pred_boxes.tensor
# #             # new_score = proposal_bbox_inst.scores
# #             # new_class = proposal_bbox_inst.pred_classes
# #             scores,index = new_score.sort(descending = True)
# #             keep_inds = []
# #             while(len(index) > 0):
# #                 cur_inx = index[0]
# #                 cur_score = scores[cur_inx]
# #                 if cur_score < confi_thres:
# #                     index = index[1:]
# #                     continue
# #                 keep = True
# #                 for ind in keep_inds:
# #                     current_bbox = new_bbox_loc[cur_inx]
# #                     remain_box = new_bbox_loc[ind]
# #                     # iou = 1
# #                     ioc = self.box_ioc_xyxy(current_bbox,remain_box)
# #                     if ioc > nms_thresh:
# #                         keep = False
# #                         break
# #                 if keep:
# #                     keep_inds.append(cur_inx)
# #                 index = index[1:]
# #             # if len(keep_inds) == 0:
# #             #     valid_map = proposal_bbox_inst.scores > thres
# #             #
# #             #     # create instances containing boxes and gt_classes
# #             #     image_shape = proposal_bbox_inst.image_size
# #             #     new_proposal_inst = Instances(image_shape)
# #             #
# #             #     # create box
# #             #     new_bbox_loc = proposal_bbox_inst.pred_boxes.tensor[valid_map, :]
# #             #     new_boxes = Boxes(new_bbox_loc)
# #             #
# #             #     # add boxes to instances
# #             #     new_proposal_inst.gt_boxes = new_boxes
# #             #     new_proposal_inst.gt_classes = proposal_bbox_inst.pred_classes[valid_map]
# #             #     new_proposal_inst.scores = proposal_bbox_inst.scores[valid_map]
# #             #     for i in new_proposal_inst.scores:
# #             #         i = 0
# #             #     return new_proposal_inst
# #
# #             keep_inds = torch.tensor(keep_inds)
# #             score_nms = new_score[keep_inds.long()]
# #             # score_nms = score_nms.reshape(-1,1)
# #             # score_nms = score_nms.reshape(-1)
# #             box_nms = new_bbox_loc[keep_inds.long()]
# #             box_nms = box_nms.reshape(-1,4)
# #             box_nms = Boxes(box_nms)
# #             class_nms = new_class[keep_inds.long()]
# #             # class_nms = class_nms.reshape(-1,1)
# #             new_proposal_inst.gt_boxes = box_nms
# #             new_proposal_inst.gt_classes = class_nms
# #             new_proposal_inst.scores = score_nms
# #
# #         elif proposal_type == "rpn":
# #
# #             raise ValueError("Unknown NMS branches")
# #
# #         return new_proposal_inst
# #
# #     def process_pseudo_label(
# #             self, proposals_rpn_unsup_k, cur_threshold, proposal_type, psedo_label_method=""
# #     ):
# #         list_instances = []
# #         num_proposal_output = 0.0
# #         for proposal_bbox_inst in proposals_rpn_unsup_k:
# #             # thresholding
# #             if psedo_label_method == "thresholding":
# #                 proposal_bbox_inst = self.threshold_bbox(
# #                     proposal_bbox_inst, thres=cur_threshold, proposal_type=proposal_type
# #                 )
# #             elif psedo_label_method == "NMS":
# #                 proposal_bbox_inst = self.NonMaxSuppression(
# #                     proposal_bbox_inst, confi_thres=cur_threshold, proposal_type=proposal_type
# #                 )
# #
# #             else:
# #                 raise ValueError("Unkown pseudo label boxes methods")
# #             num_proposal_output += len(proposal_bbox_inst)
# #             list_instances.append(proposal_bbox_inst)
# #         num_proposal_output = num_proposal_output / len(proposals_rpn_unsup_k)
# #         return list_instances, num_proposal_output
# #
# #     def remove_label(self, label_data):
# #         for label_datum in label_data:
# #             if "instances" in label_datum.keys():
# #                 del label_datum["instances"]
# #         return label_data
# #
# #     def add_label(self, unlabled_data, label):
# #         for unlabel_datum, lab_inst in zip(unlabled_data, label):
# #             unlabel_datum["instances"] = lab_inst
# #         return unlabled_data
# #
# #     def get_label(self, label_data):
# #         label_list = []
# #         for label_datum in label_data:
# #             if "instances" in label_datum.keys():
# #                 label_list.append(copy.deepcopy(label_datum["instances"]))
# #
# #         return label_list
# #     # def consistency_compare(self,pesudo_proposals_roih_unsup_k, gt_proposal, cur_compare_threshold):
# #     #     consistency = 0.5
# #     #
# #     #
# #     #     return consistency
# #
# #     # def consistency_compare(self, roi_preds, gt_label, threshold):
# #
# #
# #     # def get_label_test(self, label_data):
# #     #     label_list = []
# #     #     for label_datum in label_data:
# #     #         if "instances" in label_datum.keys():
# #     #             label_list.append(label_datum["instances"])
# #
# #     # =====================================================
# #     # =================== Training Flow ===================
# #     # =====================================================
# #
# #
# #     # def MixupDetection(self,img1,img2,label1,label2,lambd):
# #     #     # mixup two images
# #     #     height = max(img1.shape[0], img2.shape[0])
# #     #     width = max(img1.shape[1], img2.shape[1])
# #     #     mix_img = mx.nd.zeros(shape=(height, width, 3), dtype='float32')
# #     #     mix_img[:img1.shape[0], :img1.shape[1], :] = img1.astype('float32') * lambd
# #     #     mix_img[:img2.shape[0], :img2.shape[1], :] += img2.astype('float32') * (1. - lambd)
# #     #     mix_img = mix_img.astype('uint8')
# #     #     y1 = np.hstack((label1, np.full((label1.shape[0], 1), lambd)))
# #     #     y2 = np.hstack((label2, np.full((label2.shape[0], 1), 1. - lambd)))
# #     #     mix_label = np.vstack((y1, y2))
# #     #     return mix_img, mix_label
# #
# #     def run_step_full_semisup(self,teacher_ds,teacher_dp):
# #         self._trainer.iter = self.iter
# #         assert self.model.training, "[UBTeacherTrainer] model was changed to eval mode!"
# #         start = time.perf_counter()
# #         data = next(self._trainer._data_loader_iter)
# #         # data_q and data_k from different augmentations (q:strong, k:weak)
# #         # label_strong, label_weak, unlabed_strong, unlabled_weak
# #         label_train_data_q, label_train_data_k, label_compare_data_q, label_compare_data_k, unlabel_data_q, unlabel_data_k = data
# #         data_time = time.perf_counter() - start
# #         name = 520
# #         #先不更新看看情况
# #         if self.iter == self.cfg.SEMISUPNET.BURN_UP_STEP:
# #                 # update copy the the whole model
# #                 self._update_teacher_model(keep_rate=0.00)
# #                 # self.model.build_discriminator()
# #         #
# #         elif (self.iter - self.cfg.SEMISUPNET.BURN_UP_STEP
# #             ) % self.cfg.SEMISUPNET.TEACHER_UPDATE_ITER == 0:
# #                 self._update_teacher_model(
# #                     keep_rate=self.cfg.SEMISUPNET.EMA_KEEP_RATE)
# #
# #             # self._update_teacher_ds_model(teacher_ds,
# #             #     keep_rate=self.cfg.SEMISUPNET.EMA_KEEP_RATE)
# #         #     self._update_teacher_dp_model(teacher_dp,
# #         #         keep_rate=self.cfg.SEMISUPNET.EMA_KEEP_RATE)
# #
# #         record_dict = {}
# #
# #         ######################## For probe #################################
# #         # import pdb; pdb. set_trace()
# #         gt_unlabel_k = self.get_label(unlabel_data_k)
# #         # gt_unlabel_q = self.get_label_test(unlabel_data_q)
# #
# #         #  0. remove unlabeled data labels
# #         unlabel_data_q = self.remove_label(unlabel_data_q)
# #         unlabel_data_k = self.remove_label(unlabel_data_k)
# #         # fake_src_data_q = self.remove_label(label_compare_data_q)
# #         # fake_src_data_k = self.remove_label(label_compare_data_k)
# #
# #         #  1. generate the pseudo-label using teacher model
# #         # with torch.no_grad():
# #         #     (
# #         #         _,
# #         #         proposals_rpn_unsup_k,
# #         #         proposals_roih_unsup_k,
# #         #         _,
# #         #     ) = self.model_teacher(unlabel_data_k, branch="unsup_data_weak")
# #         with torch.no_grad():
# #             (
# #                 _,
# #                 proposals_rpn_unsup_k,
# #                 proposals_roih_unsup_k,
# #                 _,
# #             ) = teacher_ds(unlabel_data_k, branch="unsup_data_weak")
# #             (
# #                 _,
# #                 proposals_rpn_unsup_k_p,
# #                 proposals_roih_unsup_k_p,
# #                 _,
# #             ) = teacher_dp(unlabel_data_k, branch="unsup_data_weak")
# #         #
# #         # # todo:pseudo label fusion
# #         pesudo_proposals_roih_unsup_k,_ = self.Knowlegde_Fusion(
# #             proposals_roih_unsup_k,proposals_roih_unsup_k_p, self.cfg.SEMISUPNET.FUSION_IOU_THR,self.cfg.SEMISUPNET.FUSION_BBOX_THRESHOLD,self.cfg.SEMISUPNET.FUSION_WEIGHT
# #         )
# #
# #         ######################## For probe #################################
# #         # import pdb; pdb. set_trace()
# #
# #         # probe_metrics = ['compute_fp_gtoutlier', 'compute_num_box']
# #         # probe_metrics = ['compute_num_box']
# #         # analysis_pred, _ = self.probe.compute_num_box(gt_unlabel_k,proposals_roih_unsup_k,'pred')
# #         # record_dict.update(analysis_pred)
# #         ######################## For probe END #################################
# #
# #         #  2. Pseudo-labeling
# #         cur_threshold = self.cfg.SEMISUPNET.BBOX_THRESHOLD
# #
# #         joint_proposal_dict = {}
# #         joint_proposal_dict["proposals_rpn"] = proposals_rpn_unsup_k
# #         # Process pseudo labels and thresholding
# #         (
# #             pesudo_proposals_rpn_unsup_k,
# #             nun_pseudo_bbox_rpn,
# #         ) = self.process_pseudo_label(
# #             proposals_rpn_unsup_k, cur_threshold, "rpn", "thresholding"
# #         )
# #         # analysis_pred, _ = self.probe.compute_num_box(gt_unlabel_k,pesudo_proposals_rpn_unsup_k,'pred',True)
# #         # record_dict.update(analysis_pred)
# #
# #         joint_proposal_dict["proposals_pseudo_rpn"] = pesudo_proposals_rpn_unsup_k
# #         # Pseudo_labeling for ROI head (bbox location/objectness)
# #         #todo:123456
# #         # pesudo_proposals_roih_unsup_k, _ = self.process_pseudo_label(
# #         #     proposals_roih_unsup_k, cur_threshold, "roih", "thresholding"
# #         # )
# #
# #         joint_proposal_dict["proposals_pseudo_roih"] = pesudo_proposals_roih_unsup_k
# #         # ins_proposal_dict = {}
# #         joint_proposal_dict["proposals_pseudo"] = proposals_roih_unsup_k
# #
# #         #todo:quality eval
# #         # for i in range(len(pesudo_proposals_roih_unsup_k)):
# #         #     temp = pesudo_proposals_roih_unsup_k[i]
# #         #     # mAP_per_image = get_mAP(temp.gt_boxes,gt_unlabel_k.gt_boxes)
# #         #     recall_per, preci_per, mIOU_per = get_mIOU(temp, gt_unlabel_k[i])
# #         # #只有0和1啊，单张图像，感觉。。没什么意义啊
# #         #     recall_list.append(recall_per)
# #         #     precise_list.append(preci_per)
# #         #     mIOU_list.append(mIOU_per)
# #
# #
# #         # 3. add pseudo-label to unlabeled data
# #
# #         unlabel_data_q = self.add_label(
# #             unlabel_data_q, joint_proposal_dict["proposals_pseudo_roih"]
# #         )
# #         # unlabel_data_k = self.add_label(
# #         #     unlabel_data_k, joint_proposal_dict["proposals_pseudo_roih"]
# #         # )
# #         unlabel_data_k = self.add_label(
# #             unlabel_data_k, joint_proposal_dict["proposals_pseudo_roih"]
# #         )
# #
# #         all_label_data = label_train_data_q + label_train_data_k
# #
# #         # 4. input both strongly and weakly augmented labeled data into student model
# #         record_all_label_data, _, _, _ = self.model_teacher(
# #             all_label_data, branch="supervised"
# #         )
# #         record_dict.update(record_all_label_data)
# #
# #         all_unlabel_data = unlabel_data_q
# #         psu_sum = 0
# #         pseudo_data = []
# #         for i in range(len(unlabel_data_q)):
# #             data = unlabel_data_q[i]
# #             # print(len(data['instances']))
# #             if len(data['instances'])!=0:
# #
# #                 pseudo_data.append(data)
# #             else:
# #                 # pseudo_data.append(label_compare_data_k[i])
# #                 psu_sum +=1
# #
# #         record_all_unlabel_data, _, _, _ = self.model_teacher(
# #             all_unlabel_data, branch="supervised_target"
# #         )
# #         # record_all_unlabel_data, _, _, _ = self.model(
# #         #     all_unlabel_data, branch="supervised_target"
# #         # )
# #
# #
# #         new_record_all_unlabel_data = {}
# #         for key in record_all_unlabel_data.keys():
# #             new_record_all_unlabel_data[key + "_pseudo"] = record_all_unlabel_data[
# #                 key
# #             ]
# #         record_dict.update(new_record_all_unlabel_data)
# #
# #         # 6. input weakly labeled data (source) and weakly unlabeled data (target) to student model
# #         # give sign to the target data
# #         # unlabel_data_k =
# #
# #         for i_index in range(len(unlabel_data_k)):
# #             # unlabel_data_item = {}
# #             for k, v in unlabel_data_k[i_index].items():
# #                 # label_data_k[i_index][k + "_unlabeled"] = v
# #                 label_train_data_k[i_index][k + "_unlabeled"] = v
# #             # unlabel_data_k[i_index] = unlabel_data_item
# #
# #         all_domain_data = label_train_data_k
# #         # all_domain_data = label_data_k + unlabel_data_k
# #         # record_all_domain_data = self.model(all_domain_data, branch="domain")
# #         # record_dict.update(record_all_domain_data)
# #         # for i in range(len(unlabel_data_k)):
# #         #     t_ins_list.append(t_ins_score[i])
# #         #     t_list.append(t_img_score[i])
# #         #     s_ins_list.append(s_ins_score[i])
# #         #     s_list.append(s_img_score[i])
# #         #     for level in level_diction_list:
# #         #         level_diction_list[level].append(level_diction[level][i])
# #         #         source_level_diction_list[level].append(source_level_diction[level][i])
# #
# #         # weight losses
# #         loss_dict = {}
# #         for key in record_dict.keys():
# #             if key.startswith("loss"):
# #                 if key == "loss_rpn_loc_pseudo" or key == "loss_box_reg_pseudo":
# #                     # pseudo bbox regression <- 0
# #                     loss_dict[key] = record_dict[key] * 0
# #                 elif key[-6:] == "pseudo":  # unsupervised loss
# #                     loss_dict[key] = (
# #                             record_dict[key] *
# #                             self.cfg.SEMISUPNET.UNSUP_LOSS_WEIGHT
# #                     )
# #                 elif (
# #                         key == "loss_D_img_s" or key == "loss_D_img_t"
# #                 ):  # set weight for discriminator
# #                     # import pdb
# #                     # pdb.set_trace()
# #                     loss_dict[key] = record_dict[
# #                                          key] * 0  # Need to modify defaults and yaml
# #                 elif (key == "loss_D_ins_t" or key == "loss_D_ins_s"):
# #                     loss_dict[key] = record_dict[
# #                                          key] * 0
# #
# #                 else:  # supervised loss
# #                     loss_dict[key] = record_dict[key] * 1
# #
# #         losses = sum(loss_dict.values())
# #
# #         metrics_dict = loss_dict
# #         metrics_dict["name"] = name
# #         metrics_dict["data_time"] = data_time
# #
# #         self._write_metrics(metrics_dict)
# #
# #         self.optimizer.zero_grad()
# #         losses.backward()
# #         self.optimizer.step()
# #
# # #         if len(t_ins_list)%800 == 0:
# # #             assert len(t_list) == len(t_ins_list) == len(mIOU_list) == len(recall_list) == len(precise_list)
# # #
# # #             #target img_dis scores for recall/precise
# # #             t_dis = []
# # #             t_ins_dis = []
# # #             bar_recall_t = collections.Counter(t_dis)
# # #             bar_precis_t = collections.Counter(t_dis)
# # #
# # #             bar_recall_t_ins = collections.Counter(t_ins_dis)
# # #             bar_precis_t_ins = collections.Counter(t_ins_dis)
# # #             # bar_num = len(x_train)
# # #             # for i in range(0, 11):
# # #             #     bar_recall_t[i] = 0
# # #             #     bar_precis_t[i] = 0
# # #             for ele in t_list:
# # #                 t_dis.append(int(np.round(ele * 10)))
# # #
# # #             for ele in t_ins_list:
# # #                 t_ins_dis.append(int(np.round(ele * 10)))
# # #
# # #             len_bar = collections.Counter(t_dis)
# # #             len_ins_bar = collections.Counter(t_ins_dis)
# # #             for i in range(len(t_dis)):
# # #                 bar_recall_t[t_dis[i]] += recall_list[i]
# # #                 bar_precis_t[t_dis[i]] += precise_list[i]
# # #
# # #                 bar_recall_t_ins[t_ins_dis[i]] += recall_list[i]
# # #                 bar_precis_t_ins[t_ins_dis[i]] += precise_list[i]
# # #
# # #             pre_t_img_show = []
# # #             rec_t_img_show = []
# # #
# # #             pre_t_ins_show = []
# # #             rec_t_ins_show = []
# # #             for i in range(0,11):
# # #                 s_p = bar_precis_t[i]/(len_bar[i]+0.001)
# # #                 s_r = bar_recall_t[i]/(len_bar[i]+0.001)
# # #
# # #                 s_ins_p = bar_precis_t_ins[i] / (len_ins_bar[i] + 0.001)
# # #                 s_ins_r = bar_recall_t_ins[i] / (len_ins_bar[i] + 0.001)
# # #
# # #                 pre_t_img_show.append(s_p)
# # #                 rec_t_img_show.append(s_r)
# # #
# # #                 pre_t_ins_show.append(s_ins_p)
# # #                 rec_t_ins_show.append(s_ins_r)
# # #
# # #             #source_list_level_each
# # #             plt.figure(8)
# # #             level_show = dict(p2=[], p3=[], p4=[], p5=[], p6=[])
# # #             list_level_source = dict(p2=[], p3=[], p4=[], p5=[], p6=[])
# # #             levels = ["p2", "p3", "p4", "p5", "p6"]
# # #             for level in levels:
# # #                 for ele in source_level_diction_list[level]:
# # #                     list_level_source[level].append(int(np.round(ele * 10)))
# # #             x_axis = np.arange(11).astype(dtype=np.str)
# # #             for level in levels:
# # #                 x_train = collections.Counter(list_level_source[level])
# # #                 for i in range(0, 11):
# # #                     s = x_train[i]
# # #                     level_show[level].append(s)
# # #             plt.subplot(851)
# # #             plt.bar(x_axis, level_show["p2"], width=0.5)
# # #             plt.title('source_p2')
# # #             plt.subplot(852)
# # #             plt.bar(x_axis, level_show["p3"], width=0.5)
# # #             plt.title('source_p3')
# # #             plt.subplot(853)
# # #             plt.bar(x_axis, level_show["p4"], width=0.5)
# # #             plt.title('source_p4')
# # #             plt.subplot(854)
# # #             plt.bar(x_axis, level_show["p5"], width=0.5)
# # #             plt.title('source_p5')
# # #             plt.subplot(855)
# # #             plt.bar(x_axis, level_show["p6"], width=0.5)
# # #             plt.title('source_p6')
# # #             plt.savefig("source_level_img_bar_iter_{}.png".format(len(t_ins_list) / 4))
# # #
# # # #target_distribution
# # #             plt.figure(9)
# # #             level_show_t = dict(p2=[], p3=[], p4=[], p5=[], p6=[])
# # #             list_level_target = dict(p2=[], p3=[], p4=[], p5=[], p6=[])
# # #             for level in levels:
# # #                 for ele in level_diction_list[level]:
# # #                     list_level_target[level].append(int(np.round(ele * 10)))
# # #             for level in levels:
# # #                 x_train = collections.Counter(list_level_target[level])
# # #                 for i in range(0, 11):
# # #                     s = x_train[i]
# # #                     level_show_t[level].append(s)
# # #             plt.subplot(951)
# # #             plt.bar(x_axis, level_show_t["p2"], width=0.5)
# # #             plt.title('source_p2')
# # #             plt.subplot(952)
# # #             plt.bar(x_axis, level_show_t["p3"], width=0.5)
# # #             plt.title('source_p3')
# # #             plt.subplot(953)
# # #             plt.bar(x_axis, level_show_t["p4"], width=0.5)
# # #             plt.title('source_p4')
# # #             plt.subplot(954)
# # #             plt.bar(x_axis, level_show_t["p5"], width=0.5)
# # #             plt.title('source_p5')
# # #             plt.subplot(955)
# # #             plt.bar(x_axis, level_show_t["p6"], width=0.5)
# # #             plt.title('source_p6')
# # #             plt.savefig("target_level_img_bar_iter_{}.png".format(len(t_ins_list) / 4))
# # #
# # #
# # #             #img
# # #             plt.figure(6)
# # #             plt.subplot(621)
# # #             plt.bar(x_axis,pre_t_img_show,width=0.5)
# # #             plt.xlabel('t_img_dis')
# # #             plt.ylabel('precise_score')
# # #             plt.title('target_dis_img & precise')
# # #
# # #             plt.subplot(622)
# # #             plt.bar(x_axis, rec_t_img_show, width=0.5)
# # #             plt.xlabel('t_img_dis')
# # #             plt.ylabel('recall_score')
# # #             plt.title('target_dis_img & recall')
# # #             plt.savefig("prediction_quailty_with_domain_img_bar_iter_{}.png".format(len(t_ins_list) / 4))
# # #
# # #             #ins
# # #             plt.figure(7)
# # #             plt.subplot(721)
# # #             plt.bar(x_axis, pre_t_ins_show, width=0.5)
# # #             plt.xlabel('t_ins_dis')
# # #             plt.ylabel('precise_score')
# # #             plt.title('target_dis_ins & precise')
# # #
# # #             plt.subplot(722)
# # #             plt.bar(x_axis, rec_t_ins_show, width=0.5)
# # #             plt.xlabel('t_ins_dis')
# # #             plt.ylabel('recall_score')
# # #             plt.title('target_dis_ins & recall')
# # #             plt.savefig("prediction_quailty_with_domain_ins_bar_iter_{}.png".format(len(t_ins_list) / 4))
# # #
# # #             #scatter for miou and dis_img
# # #             plt.figure(1)
# # #
# # #
# # #             plt.subplot(133)
# # #             plt.scatter(mIOU_list, t_list,s = 1)
# # #             plt.xlabel('mIOU')
# # #             plt.ylabel('img_dis')
# # #             plt.title('mIOU & domain_img')
# # #             plt.savefig("prediction_quailty_with_domain_img_iter_{}.png".format(len(t_ins_list) / 4))
# # #             #
# # #
# # #             #scatter for miou and dis_ins
# # #             plt.figure(4)
# # #
# # #
# # #             plt.subplot(433)
# # #             plt.scatter(mIOU_list, t_ins_list,s=1)
# # #             plt.xlabel('mIOU')
# # #             plt.ylabel('ins_dis')
# # #             plt.title('iou & domain_ins')
# # #             #
# # #             plt.savefig("prediction_quailty_with_domain_ins_iter_{}.png".format(len(t_ins_list)/4))
# # #
# # #
# # #             #bar of s_ins_num and s_img_num
# # #
# # #             plt.figure(5)
# # #             plt.subplot(531)
# # #             list_ins_source = []
# # #             ppt_s_ins = []
# # #             for ele in s_ins_list:
# # #                 list_ins_source.append(int(np.round(ele * 10)))
# # #             x_train = collections.Counter(list_ins_source)
# # #             x_num = len(x_train)
# # #             x_axis = np.arange(11).astype(dtype=np.str)
# # #             for i in range(0, 11):
# # #                 s = x_train[i]
# # #                 ppt_s_ins.append(s)
# # #             plt.bar(x_axis, ppt_s_ins, width=0.5)
# # #             # plt.savefig('ins_s.png')
# # #             plt.xlabel('s_ins')
# # #             plt.ylabel('dis_num')
# # #             plt.title('source ins')
# # #
# # #             plt.subplot(532)
# # #             list_img_source = []
# # #             ppt_s_img = []
# # #             for ele in s_list:
# # #                 list_img_source.append(int(np.round(ele * 10)))
# # #             x_train = collections.Counter(list_img_source)
# # #             x_num = len(x_train)
# # #             x_axis = np.arange(11).astype(dtype=np.str)
# # #             for i in range(0, 11):
# # #                 s = x_train[i]
# # #                 ppt_s_img.append(s)
# # #             plt.bar(x_axis, ppt_s_img, width=0.5)
# # #             # plt.savefig('ins_s.png')
# # #             plt.xlabel('s_img')
# # #             plt.ylabel('dis_num')
# # #             plt.title('source img')
# # #
# # #             plt.savefig("source_iter_reiou_{}.png".format(len(t_ins_list) / 4))
# # #
# # #
# # #             #bar of target_ins_num and target_img_num
# # #             plt.figure(3)
# # #
# # #             plt.subplot(331)
# # #             list_ins_target = []
# # #             ppt_t_ins = []
# # #             for ele in t_ins_list:
# # #                 try:
# # #                     list_ins_target.append(int(np.round(ele * 10)))
# # #                 except:
# # #                     continue
# # #             x_train = collections.Counter(list_ins_target)
# # #             x_num = len(x_train)
# # #             x_axis = np.arange(11).astype(dtype=np.str)
# # #             for i in range(0, 11):
# # #                 s = x_train[i]
# # #                 ppt_t_ins.append(s)
# # #             plt.bar(x_axis, ppt_t_ins, width=0.5)
# # #             # plt.savefig('ins_s.png')
# # #             plt.xlabel('t_ins')
# # #             plt.ylabel('dis_num')
# # #             plt.title('target ins')
# # #
# # #             plt.subplot(332)
# # #             list_img_target = []
# # #             ppt_t_img = []
# # #             for ele in t_list:
# # #                 list_img_target.append(int(np.round(ele * 10)))
# # #             x_train = collections.Counter(list_img_target)
# # #             x_num = len(x_train)
# # #             x_axis = np.arange(11).astype(dtype=np.str)
# # #             for i in range(0, 11):
# # #                 s = x_train[i]
# # #                 ppt_t_img.append(s)
# # #             plt.bar(x_axis, ppt_t_img, width=0.5)
# # #             # plt.savefig('ins_s.png')
# # #             plt.xlabel('t_img')
# # #             plt.ylabel('dis_num')
# # #             plt.title('target img')
# # #
# # #             plt.savefig("target_iter_reiou_{}.png".format(len(t_ins_list) / 4))
# #
# #
# #
# #             # plt.subplot(331)
# #             # plt.scatter(s_ins_list, s_list, s=1)
# #             # plt.xlabel('S_ins')
# #             # plt.ylabel('S_img')
# #             # plt.title('source ins & img')
# #
# #             # plt.subplot(332)
# #             # plt.scatter(t_ins_list, t_list, s=1)
# #             # plt.xlabel('t_ins')
# #             # plt.ylabel('t_img')
# #             # plt.title('target ins & img')
# #
# #
# #             # plt.savefig("source_iter_{}.png".format(len(t_ins_list) / 4))
# #
# #         # return psu_sum
# #
# #         # self._trainer.iter = self.iter
# #         # assert self.model.training, "[UBTeacherTrainer] model was changed to eval mode!"
# #         # start = time.perf_counter()
# #         # data = next(self._trainer._data_loader_iter)
# #         # # data_q and data_k from different augmentations (q:strong, k:weak)
# #         # # label_strong, label_weak, unlabed_strong, unlabled_weak
# #         # label_data_q, label_data_k,label_compare_data_q,label_compare_data_k, unlabel_data_q, unlabel_data_k = data
# #         # data_time = time.perf_counter() - start
# #         #
# #         # # burn-in stage (supervised training with labeled data)
# #         # if self.iter < self.cfg.SEMISUPNET.BURN_UP_STEP:
# #         #
# #         #     # input both strong and weak supervised data into model
# #         #     label_data_q.extend(label_data_k)
# #         #     record_dict, _, _, _ = self.model(
# #         #         label_data_q, branch="supervised")
# #         #
# #         #     # weight losses
# #         #     loss_dict = {}
# #         #     for key in record_dict.keys():
# #         #         if key[:4] == "loss":
# #         #             loss_dict[key] = record_dict[key] * 1
# #         #     losses = sum(loss_dict.values())
# #         #
# #         # else:
# #         #     if self.iter == self.cfg.SEMISUPNET.BURN_UP_STEP:
# #         #         # update copy the the whole model
# #         #         self._update_teacher_model(keep_rate=0.00)
# #         #         # self.model.build_discriminator()
# #         #
# #         #     elif (
# #         #         self.iter - self.cfg.SEMISUPNET.BURN_UP_STEP
# #         #     ) % self.cfg.SEMISUPNET.TEACHER_UPDATE_ITER == 0:
# #         #         self._update_teacher_model(
# #         #             keep_rate=self.cfg.SEMISUPNET.EMA_KEEP_RATE)
# #         #
# #         #     record_dict = {}
# #         #
# #         #     ######################## For probe #################################
# #         #     # import pdb; pdb. set_trace()
# #         #     gt_unlabel_k = self.get_label(unlabel_data_k)
# #         #     # gt_unlabel_q = self.get_label_test(unlabel_data_q)
# #         #
# #         #
# #         #     #  0. remove unlabeled data labels
# #         #     unlabel_data_q = self.remove_label(unlabel_data_q)
# #         #     unlabel_data_k = self.remove_label(unlabel_data_k)
# #         #
# #         #     #  1. generate the pseudo-label using teacher model
# #         #     with torch.no_grad():
# #         #         (
# #         #             _,
# #         #             proposals_rpn_unsup_k,
# #         #             proposals_roih_unsup_k,
# #         #             _,
# #         #         ) = self.model_teacher(unlabel_data_k, branch="unsup_data_weak")
# #         #
# #         #     ######################## For probe #################################
# #         #     # import pdb; pdb. set_trace()
# #         #
# #         #     # probe_metrics = ['compute_fp_gtoutlier', 'compute_num_box']
# #         #     # probe_metrics = ['compute_num_box']
# #         #     # analysis_pred, _ = self.probe.compute_num_box(gt_unlabel_k,proposals_roih_unsup_k,'pred')
# #         #     # record_dict.update(analysis_pred)
# #         #     ######################## For probe END #################################
# #         #
# #         #     #  2. Pseudo-labeling
# #         #     cur_threshold = self.cfg.SEMISUPNET.BBOX_THRESHOLD
# #         #
# #         #     joint_proposal_dict = {}
# #         #     joint_proposal_dict["proposals_rpn"] = proposals_rpn_unsup_k
# #         #     #Process pseudo labels and thresholding
# #         #     (
# #         #         pesudo_proposals_rpn_unsup_k,
# #         #         nun_pseudo_bbox_rpn,
# #         #     ) = self.process_pseudo_label(
# #         #         proposals_rpn_unsup_k, cur_threshold, "rpn", "thresholding"
# #         #     )
# #         #     # analysis_pred, _ = self.probe.compute_num_box(gt_unlabel_k,pesudo_proposals_rpn_unsup_k,'pred',True)
# #         #     # record_dict.update(analysis_pred)
# #         #
# #         #     joint_proposal_dict["proposals_pseudo_rpn"] = pesudo_proposals_rpn_unsup_k
# #         #     # Pseudo_labeling for ROI head (bbox location/objectness)
# #         #     pesudo_proposals_roih_unsup_k, _ = self.process_pseudo_label(
# #         #         proposals_roih_unsup_k, cur_threshold, "roih", "thresholding"
# #         #     )
# #         #     joint_proposal_dict["proposals_pseudo_roih"] = pesudo_proposals_roih_unsup_k
# #         #
# #         #     # 3. add pseudo-label to unlabeled data
# #         #
# #         #     unlabel_data_q = self.add_label(
# #         #         unlabel_data_q, joint_proposal_dict["proposals_pseudo_roih"]
# #         #     )
# #         #     unlabel_data_k = self.add_label(
# #         #         unlabel_data_k, joint_proposal_dict["proposals_pseudo_roih"]
# #         #     )
# #         #
# #         #     all_label_data = label_data_q + label_data_k
# #         #     all_unlabel_data = unlabel_data_q
# #         #
# #         #     # 4. input both strongly and weakly augmented labeled data into student model
# #         #     record_all_label_data, _, _, _ = self.model(
# #         #         all_label_data, branch="supervised"
# #         #     )
# #         #     record_dict.update(record_all_label_data)
# #         #
# #         #     # 5. input strongly augmented unlabeled data into model
# #         #     record_all_unlabel_data, _, _, _ = self.model(
# #         #         all_unlabel_data, branch="supervised_target"
# #         #     )
# #         #     new_record_all_unlabel_data = {}
# #         #     for key in record_all_unlabel_data.keys():
# #         #         new_record_all_unlabel_data[key + "_pseudo"] = record_all_unlabel_data[
# #         #             key
# #         #         ]
# #         #     record_dict.update(new_record_all_unlabel_data)
# #         #
# #         #     # 6. input weakly labeled data (source) and weakly unlabeled data (target) to student model
# #         #     # give sign to the target data
# #         #
# #         #     for i_index in range(len(unlabel_data_k)):
# #         #         # unlabel_data_item = {}
# #         #         for k, v in unlabel_data_k[i_index].items():
# #         #             # label_data_k[i_index][k + "_unlabeled"] = v
# #         #             label_data_k[i_index][k + "_unlabeled"] = v
# #         #         # unlabel_data_k[i_index] = unlabel_data_item
# #         #
# #         #     all_domain_data = label_data_k
# #         #     # all_domain_data = label_data_k + unlabel_data_k
# #         #     record_all_domain_data, _, _, _ = self.model(all_domain_data, branch="domain")
# #         #     record_dict.update(record_all_domain_data)
# #         #
# #         #
# #         #     # weight losses
# #         #     loss_dict = {}
# #         #     for key in record_dict.keys():
# #         #         if key.startswith("loss"):
# #         #             if key == "loss_rpn_loc_pseudo" or key == "loss_box_reg_pseudo":
# #         #                 # pseudo bbox regression <- 0
# #         #                 loss_dict[key] = record_dict[key] * 0
# #         #             elif key[-6:] == "pseudo":  # unsupervised loss
# #         #                 loss_dict[key] = (
# #         #                     record_dict[key] *
# #         #                     self.cfg.SEMISUPNET.UNSUP_LOSS_WEIGHT
# #         #                 )
# #         #             elif (
# #         #                 key == "loss_D_img_s" or key == "loss_D_img_t"
# #         #             ):  # set weight for discriminator
# #         #                 # import pdb
# #         #                 # pdb.set_trace()
# #         #                 loss_dict[key] = record_dict[key] * self.cfg.SEMISUPNET.DIS_LOSS_WEIGHT #Need to modify defaults and yaml
# #         #             else:  # supervised loss
# #         #                 loss_dict[key] = record_dict[key] * 1
# #         #
# #         #     losses = sum(loss_dict.values())
# #         #
# #         # metrics_dict = record_dict
# #         # metrics_dict["data_time"] = data_time
# #         # self._write_metrics(metrics_dict)
# #         #
# #         # self.optimizer.zero_grad()
# #         # losses.backward()
# #         # self.optimizer.step()
# #
# #     def _write_metrics(self, metrics_dict: dict):
# #         metrics_dict = {
# #             k: v.detach().cpu().item() if isinstance(v, torch.Tensor) else float(v)
# #             for k, v in metrics_dict.items()
# #         }
# #
# #         # gather metrics among all workers for logging
# #         # This assumes we do DDP-style training, which is currently the only
# #         # supported method in detectron2.
# #         all_metrics_dict = comm.gather(metrics_dict)
# #         # all_hg_dict = comm.gather(hg_dict)
# #
# #         if comm.is_main_process():
# #             if "data_time" in all_metrics_dict[0]:
# #                 # data_time among workers can have high variance. The actual latency
# #                 # caused by data_time is the maximum among workers.
# #                 data_time = np.max([x.pop("data_time")
# #                                     for x in all_metrics_dict])
# #                 self.storage.put_scalar("data_time", data_time)
# #
# #             # average the rest metrics
# #             metrics_dict = {
# #                 k: np.mean([x[k] for x in all_metrics_dict])
# #                 for k in all_metrics_dict[0].keys()
# #             }
# #
# #             # append the list
# #             loss_dict = {}
# #             for key in metrics_dict.keys():
# #                 if key[:4] == "loss":
# #                     loss_dict[key] = metrics_dict[key]
# #
# #             total_losses_reduced = sum(loss for loss in loss_dict.values())
# #
# #             self.storage.put_scalar("total_loss", total_losses_reduced)
# #             if len(metrics_dict) > 1:
# #                 self.storage.put_scalars(**metrics_dict)
# #     @torch.no_grad()
# #     def _update_student_model(self, keep_rate=0.999):
# #         if comm.get_world_size() > 1:
# #             student_model_dict = {
# #                 key[7:]: value for key, value in self.model_teacher.state_dict().items()
# #             }
# #         else:
# #             student_model_dict = self.model_teacher.state_dict()
# #
# #         new_teacher_dict = OrderedDict()
# #         for key, value in self.model.state_dict().items():
# #             if key in student_model_dict.keys():
# #                 new_teacher_dict[key] = (
# #                         student_model_dict[key] *
# #                         (1 - keep_rate) + value * keep_rate
# #                 )
# #             else:
# #                 raise Exception("{} is not found in student model".format(key))
# #
# #         self.model.load_state_dict(new_teacher_dict)
# #     @torch.no_grad()
# #     def _update_teacher_model(self, keep_rate=0.999):
# #         if comm.get_world_size() > 1:
# #             student_model_dict = {
# #                 key[7:]: value for key, value in self.model.state_dict().items()
# #             }
# #         else:
# #             student_model_dict = self.model.state_dict()
# #
# #         new_teacher_dict = OrderedDict()
# #         for key, value in self.model_teacher.state_dict().items():
# #             if key in student_model_dict.keys():
# #                 new_teacher_dict[key] = (
# #                         student_model_dict[key] *
# #                         (1 - keep_rate) + value * keep_rate
# #                 )
# #             else:
# #                 raise Exception("{} is not found in student model".format(key))
# #
# #         self.model_teacher.load_state_dict(new_teacher_dict)
# #
# #     @torch.no_grad()
# #     def _update_teacher_ds_model(self, model_ds,keep_rate=0.9996):
# #         if comm.get_world_size() > 1:
# #             student_model_dict = {
# #                 key[7:]: value for key, value in self.model.state_dict().items()
# #             }
# #         else:
# #             student_model_dict = self.model.state_dict()
# #
# #         new_teacher_dict = OrderedDict()
# #         for key, value in model_ds.state_dict().items():
# #             if key in student_model_dict.keys():
# #                 new_teacher_dict[key] = (
# #                         student_model_dict[key] *
# #                         (1 - keep_rate) + value * keep_rate
# #                 )
# #             else:
# #                 raise Exception("{} is not found in student model".format(key))
# #
# #         model_ds.load_state_dict(new_teacher_dict)
# #
# #     @torch.no_grad()
# #     def _update_teacher_dp_model(self,model_dp, keep_rate=0.9996):
# #         if comm.get_world_size() > 1:
# #             student_model_dict = {
# #                 key[7:]: value for key, value in self.model.state_dict().items()
# #             }
# #         else:
# #             student_model_dict = self.model.state_dict()
# #
# #         new_teacher_dict = OrderedDict()
# #         for key, value in model_dp.state_dict().items():
# #             if key in student_model_dict.keys():
# #                 new_teacher_dict[key] = (
# #                         student_model_dict[key] *
# #                         (1 - keep_rate) + value * keep_rate
# #                 )
# #             else:
# #                 raise Exception("{} is not found in student model".format(key))
# #
# #         model_dp.load_state_dict(new_teacher_dict)
# #
# #     @torch.no_grad()
# #     def _copy_main_model(self):
# #         # initialize all parameters
# #         if comm.get_world_size() > 1:
# #             rename_model_dict = {
# #                 key[7:]: value for key, value in self.model.state_dict().items()
# #             }
# #             self.model_teacher.load_state_dict(rename_model_dict)
# #         else:
# #             self.model_teacher.load_state_dict(self.model.state_dict())
# #
# #     @classmethod
# #     def build_test_loader(cls, cfg, dataset_name):
# #         return build_detection_test_loader(cfg, dataset_name)
# #
# #     def build_hooks(self):
# #         cfg = self.cfg.clone()
# #         cfg.defrost()
# #         cfg.DATALOADER.NUM_WORKERS = 0  # save some memory and time for PreciseBN
# #
# #         ret = [
# #             hooks.IterationTimer(),
# #             hooks.LRScheduler(self.optimizer, self.scheduler),
# #             hooks.PreciseBN(
# #                 # Run at the same freq as (but before) evaluation.
# #                 cfg.TEST.EVAL_PERIOD,
# #                 self.model,
# #                 # Build a new data loader to not affect training
# #                 self.build_train_loader(cfg),
# #                 cfg.TEST.PRECISE_BN.NUM_ITER,
# #             )
# #             if cfg.TEST.PRECISE_BN.ENABLED and get_bn_modules(self.model)
# #             else None,
# #         ]
# #
# #         # Do PreciseBN before checkpointer, because it updates the model and need to
# #         # be saved by checkpointer.
# #         # This is not always the best: if checkpointing has a different frequency,
# #         # some checkpoints may have more precise statistics than others.
# #         if comm.is_main_process():
# #             ret.append(
# #                 hooks.PeriodicCheckpointer(
# #                     self.checkpointer, cfg.SOLVER.CHECKPOINT_PERIOD
# #                 )
# #             )
# #
# #         def test_and_save_results_student():
# #             self._last_eval_results_student = self.test(self.cfg, self.model)
# #             _last_eval_results_student = {
# #                 k + "_student": self._last_eval_results_student[k]
# #                 for k in self._last_eval_results_student.keys()
# #             }
# #             return _last_eval_results_student
# #
# #         def test_and_save_results_teacher():
# #             self._last_eval_results_teacher = self.test(
# #                 self.cfg, self.model_teacher)
# #             return self._last_eval_results_teacher
# #
# #         ret.append(hooks.EvalHook(cfg.TEST.EVAL_PERIOD,
# #                                   test_and_save_results_student))
# #         ret.append(hooks.EvalHook(cfg.TEST.EVAL_PERIOD,
# #                                   test_and_save_results_teacher))
# #
# #         if comm.is_main_process():
# #             # run writers in the end, so that evaluation metrics are written
# #             ret.append(hooks.PeriodicWriter(self.build_writers(), period=20))
# #         return ret




###########################################################################################################################

import os
import time
import logging
import torch
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.parallel import DistributedDataParallel
from fvcore.nn.precise_bn import get_bn_modules
import numpy as np
from collections import OrderedDict
# from matplotlib.font_manager import FontProperties
import cv2
# import numpy as np
import detectron2.utils.comm as comm
from detectron2.checkpoint import DetectionCheckpointer
from detectron2.engine import DefaultTrainer, SimpleTrainer, TrainerBase
from detectron2.engine.train_loop import AMPTrainer
from detectron2.utils.events import EventStorage
from detectron2.evaluation import verify_results, DatasetEvaluators
# from detectron2.evaluation import COCOEvaluator, verify_results, DatasetEvaluators

from detectron2.data.dataset_mapper import DatasetMapper
from detectron2.engine import hooks
from detectron2.structures.boxes import Boxes
from detectron2.structures.instances import Instances
from detectron2.utils.env import TORCH_VERSION
from detectron2.data import MetadataCatalog

from adapteacher.data.build import (
    build_detection_semisup_train_loader,
    build_detection_test_loader,
    build_detection_semisup_train_loader_two_crops,
)
from adapteacher.data.dataset_mapper import DatasetMapperTwoCropSeparate
from adapteacher.engine.hooks import LossEvalHook
# from adapteacher.modeling.meta_arch.ts_ensemble import EnsembleTSModel
from adapteacher.checkpoint.detection_checkpoint import DetectionTSCheckpointer
from adapteacher.solver.build import build_lr_scheduler
from adapteacher.evaluation import PascalVOCDetectionEvaluator, COCOEvaluator
# from utils.fft import FDA_source_to_target_np
from .probe import OpenMatchTrainerProbe
import copy
import collections
from torch.utils.tensorboard import SummaryWriter


# Supervised-only Trainer
class BaselineTrainer(DefaultTrainer):
    def __init__(self, cfg):
        """
        Args:
            cfg (CfgNode):
        Use the custom checkpointer, which loads other backbone models
        with matching heuristics.
        """
        cfg = DefaultTrainer.auto_scale_workers(cfg, comm.get_world_size())
        model = self.build_model(cfg)
        optimizer = self.build_optimizer(cfg, model)
        data_loader = self.build_train_loader(cfg)

        if comm.get_world_size() > 1:
            model = DistributedDataParallel(
                model, device_ids=[comm.get_local_rank()], broadcast_buffers=False
            )

        TrainerBase.__init__(self)
        self._trainer = (AMPTrainer if cfg.SOLVER.AMP.ENABLED else SimpleTrainer)(
            model, data_loader, optimizer
        )

        self.scheduler = self.build_lr_scheduler(cfg, optimizer)
        self.checkpointer = DetectionCheckpointer(
            model,
            cfg.OUTPUT_DIR,
            optimizer=optimizer,
            scheduler=self.scheduler,
        )
        self.start_iter = 0
        self.start_round = 0
        self.max_iter = cfg.SOLVER.MAX_ITER
        self.max_round = cfg.SOLVER.MAX_ROUND
        self.cfg = cfg

        self.register_hooks(self.build_hooks())

    def resume_or_load(self, resume=True):
        """
        If `resume==True` and `cfg.OUTPUT_DIR` contains the last checkpoint (defined by
        a `last_checkpoint` file), resume from the file. Resuming means loading all
        available states (eg. optimizer and scheduler) and update iteration counter
        from the checkpoint. ``cfg.MODEL.WEIGHTS`` will not be used.
        Otherwise, this is considered as an independent training. The method will load model
        weights from the file `cfg.MODEL.WEIGHTS` (but will not load other states) and start
        from iteration 0.
        Args:
            resume (bool): whether to do resume or not
        """
        checkpoint = self.checkpointer.resume_or_load(
            self.cfg.MODEL.WEIGHTS, resume=resume
        )
        if resume and self.checkpointer.has_checkpoint():
            self.start_iter = checkpoint.get("iteration", -1) + 1
            # The checkpoint stores the training iteration that just finished, thus we start
            # at the next iteration (or iter zero if there's no checkpoint).
        if isinstance(self.model, DistributedDataParallel):
            # broadcast loaded data/model from the first rank, because other
            # machines may not have access to the checkpoint file
            if TORCH_VERSION >= (1, 7):
                self.model._sync_params_and_buffers()
            self.start_iter = comm.all_gather(self.start_iter)[0]

    def train_loop(self, start_iter: int, max_iter: int,start_round: int,max_round:int ):
        """
        Args:
            start_iter, max_iter (int): See docs above
        """
        logger = logging.getLogger(__name__)
        logger.info("Starting training from iteration {}".format(start_iter))
        self.round = self.start_round = start_round
        self.iter = self.start_iter = start_iter
        self.max_iter = max_iter
        self.max_round = max_round

        with EventStorage(start_iter) as self.storage:
            try:
                self.before_train()
                for self.iter in range(start_iter, max_iter):
                    self.before_step()
                    self.run_step()
                    self.after_step()
            except Exception:
                logger.exception("Exception during training:")
                raise
            finally:
                self.after_train()

    def run_step(self):
        self._trainer.iter = self.iter

        assert self.model.training, "[SimpleTrainer] model was changed to eval mode!"
        start = time.perf_counter()

        data = next(self._trainer._data_loader_iter)
        data_time = time.perf_counter() - start

        record_dict, _, _, _ = self.model(data, branch="supervised")

        num_gt_bbox = 0.0
        for element in data:
            num_gt_bbox += len(element["instances"])
        num_gt_bbox = num_gt_bbox / len(data)
        record_dict["bbox_num/gt_bboxes"] = num_gt_bbox

        loss_dict = {}
        for key in record_dict.keys():
            if key[:4] == "loss" and key[-3:] != "val":
                loss_dict[key] = record_dict[key]

        losses = sum(loss_dict.values())

        metrics_dict = record_dict
        metrics_dict["data_time"] = data_time
        self._write_metrics(metrics_dict)

        self.optimizer.zero_grad()
        losses.backward()
        self.optimizer.step()

    @classmethod
    def build_evaluator(cls, cfg, dataset_name, output_folder=None):
        if output_folder is None:
            output_folder = os.path.join(cfg.OUTPUT_DIR, "inference")
        evaluator_list = []
        evaluator_type = MetadataCatalog.get(dataset_name).evaluator_type

        if evaluator_type == "coco":
            evaluator_list.append(COCOEvaluator(
                dataset_name, output_dir=output_folder))
        elif evaluator_type == "pascal_voc":
            return PascalVOCDetectionEvaluator(dataset_name)
        elif evaluator_type == "pascal_voc_water":
            return PascalVOCDetectionEvaluator(dataset_name,
                                               target_classnames=["bicycle", "bird", "car", "cat", "dog", "person"])
        if len(evaluator_list) == 0:
            raise NotImplementedError(
                "no Evaluator for the dataset {} with the type {}".format(
                    dataset_name, evaluator_type
                )
            )
        elif len(evaluator_list) == 1:
            return evaluator_list[0]

        return DatasetEvaluators(evaluator_list)

    @classmethod
    def build_train_loader(cls, cfg):
        return build_detection_semisup_train_loader(cfg, mapper=None)

    @classmethod
    def build_test_loader(cls, cfg, dataset_name):
        """
        Returns:
            iterable
        """
        return build_detection_test_loader(cfg, dataset_name)

    def build_hooks(self):
        """
        Build a list of default hooks, including timing, evaluation,
        checkpointing, lr scheduling, precise BN, writing events.

        Returns:
            list[HookBase]:
        """
        cfg = self.cfg.clone()
        cfg.defrost()
        cfg.DATALOADER.NUM_WORKERS = 0

        ret = [
            hooks.IterationTimer(),
            hooks.LRScheduler(self.optimizer, self.scheduler),
            hooks.PreciseBN(
                cfg.TEST.EVAL_PERIOD,
                self.model,
                self.build_train_loader(cfg),
                cfg.TEST.PRECISE_BN.NUM_ITER,
            )
            if cfg.TEST.PRECISE_BN.ENABLED and get_bn_modules(self.model)
            else None,
        ]

        if comm.is_main_process():
            ret.append(
                hooks.PeriodicCheckpointer(
                    self.checkpointer, cfg.SOLVER.CHECKPOINT_PERIOD
                )
            )

        def test_and_save_results():
            self._last_eval_results = self.test(self.cfg, self.model)
            return self._last_eval_results

        ret.append(hooks.EvalHook(cfg.TEST.EVAL_PERIOD, test_and_save_results))

        if comm.is_main_process():
            ret.append(hooks.PeriodicWriter(self.build_writers(), period=20))
        return ret

    def _write_metrics(self, metrics_dict: dict):
        """
        Args:
            metrics_dict (dict): dict of scalar metrics
        """
        metrics_dict = {
            k: v.detach().cpu().item() if isinstance(v, torch.Tensor) else float(v)
            for k, v in metrics_dict.items()
        }
        # gather metrics among all workers for logging
        # This assumes we do DDP-style training, which is currently the only
        # supported method in detectron2.
        all_metrics_dict = comm.gather(metrics_dict)

        if comm.is_main_process():
            if "data_time" in all_metrics_dict[0]:
                data_time = np.max([x.pop("data_time")
                                    for x in all_metrics_dict])
                self.storage.put_scalar("data_time", data_time)

            metrics_dict = {
                k: np.mean([x[k] for x in all_metrics_dict])
                for k in all_metrics_dict[0].keys()
            }

            loss_dict = {}
            for key in metrics_dict.keys():
                if key[:4] == "loss":
                    loss_dict[key] = metrics_dict[key]

            total_losses_reduced = sum(loss for loss in loss_dict.values())

            self.storage.put_scalar("total_loss", total_losses_reduced)
            if len(metrics_dict) > 1:
                self.storage.put_scalars(**metrics_dict)

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
            # box_area = (box_part[3]-box_part[1]) * (box_part[2]-box_part[0])
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

def get_mIOU(pred_ins, gt_ins, iou_threh = 0.5):
    pred_boxs, gt_boxs = pred_ins.gt_boxes.tensor, gt_ins.gt_boxes.tensor
    iou = np.zeros((len(pred_boxs), len(gt_boxs)))
    #calculate iou for filter
    for j in range(len(gt_boxs)):
        for i in range(len(pred_boxs)):
            iou[i][j] = bb_intersection_over_union(pred_boxs[i],gt_boxs[j])
    #filter
    iou_list = []
    gt_index = iou.argmax(axis=1)
    if len(pred_boxs)>0:
        pred_index = iou.argmax(axis=0)
        for j in range(len(gt_boxs)):
            iou_list.append(iou[pred_index][j])
    gt_index[iou.max(axis=1) < iou_threh] = -1
    select = np.zeros(gt_boxs.size(0), dtype=bool)
    match = []
    confidence = []
    for sid, gt_idx in enumerate(gt_index):
        if gt_idx >= 0:
            if not select[gt_idx]:
                match.append(1)
            else:
                match.append(0)
            select[gt_idx] = True
        else:
            match.append(0)
        # iou_list.append(iou[sid][gt_idx])
        confidence.append(pred_ins.scores[sid].detach().cpu())

    match = np.asarray(match)
    confidence = np.asarray(confidence)
    iou_list = np.asarray(iou_list)
    order = confidence.argsort()[::-1]
    match = match[order]

    # tp = np.cumsum(match == 1).sum()
    # fp = np.cumsum(match == 0).sum()
    tp = (match == 1).sum()
    fp = (match == 0).sum()

    if len(match) != 0 :

        rec = tp / (fp + tp)
        prec = tp / len(match)
        mIOU = iou_list.mean()
    else:
        rec = 0.0
        prec = 0.0
        mIOU = 0.0
#这个iou是以gt为基准，原本的是按照pred_box，如果是按面积计，他的无疑更合理
    return rec, prec, mIOU

def get_mIOU2(pred_ins, gt_ins, iou_threh = 0.5):
    pred_boxs, gt_boxs = pred_ins.gt_boxes.tensor, gt_ins.gt_boxes.tensor
    iou = np.zeros((len(pred_boxs), len(gt_boxs)))
    #calculate iou for filter
    for j in range(len(gt_boxs)):
        for i in range(len(pred_boxs)):
            iou[i][j] = bb_intersection_over_union(pred_boxs[i],gt_boxs[j])
    #filter
    iou_list = []
    gt_index = iou.argmax(axis=1)
    if len(pred_boxs)>0:
        pred_index = iou.argmax(axis=0)
        # for j in range(len(gt_boxs)):
        #     iou_list.append(iou[pred_index][j])
    gt_index[iou.max(axis=1) < iou_threh] = -1
    select = np.zeros(gt_boxs.size(0), dtype=bool)
    match = []
    confidence = []
    for sid, gt_idx in enumerate(gt_index):
        if gt_idx >= 0:
            if not select[gt_idx]:
                match.append(1)
                iou_list.append(iou[sid][gt_idx])
            else:
                match.append(0)
            select[gt_idx] = True
        else:
            match.append(0)

        confidence.append(pred_ins.scores[sid].detach().cpu())

    match = np.asarray(match)
    confidence = np.asarray(confidence)
    iou_list = np.asarray(iou_list)
    order = confidence.argsort()[::-1]
    match = match[order]

    # tp = np.cumsum(match == 1).sum()
    # fp = np.cumsum(match == 0).sum()
    tp = (match == 1).sum()
    fp = (match == 0).sum()

    rec = tp / (fp + tp)
    prec = tp / len(match)
    mIOU = iou_list.mean()
#这个是按面积计miou
    return rec, prec, mIOU

def get_weighted_box(boxes, conf_type='avg'):
    """
    Create weighted box for set of boxes
    :param boxes: set of boxes to fuse
    :param conf_type: type of confidence one of 'avg' or 'max'
    :return: weighted box
    """
    #oral
    # box = np.zeros(6, dtype=np.float32)
    # conf = 0
    # conf_list = []
    # for b in boxes:
    #     box[2:] += (b[1] * b[2:])
    #     conf += b[1]
    #     conf_list.append(b[1])
    # box[0] = boxes[0][0]
    # if conf_type == 'avg':
    #     box[1] = conf / len(boxes)
    # elif conf_type == 'max':
    #     box[1] = np.array(conf_list).max()
    # box[2:] /= conf

    #area_weights
    box = np.zeros(6, dtype=np.float32)
    conf = 0
    area = 0
    i = 0
    box_area1 = int((boxes[0][5] - boxes[0][3]) * (boxes[0][4] - boxes[0][2]))
    box_area2 = int((boxes[1][5] - boxes[1][3]) * (boxes[1][4] - boxes[1][2]))
    # area_weights = [box_area1 / box_area2,1]
    conf_list = []
    for b in boxes:
        # box_area = (b[5] -b[3])* (b[4]-b[2])
        box[2:] += (b[1] * b[2:])
        conf += b[1]
        i+=1
        conf_list.append(b[1])
    box[0] = boxes[0][0]
    if conf_type == 'avg':
        box[1] = conf / len(boxes)
    elif conf_type == 'max':
        box[1] = np.array(conf_list).max()
    box[2:] /= (conf )
    return box


grads = {}


# 这个函数是为了获取中间变量的梯度，我方案中的Z不是一个叶子结点，所以其梯度在反向传播之后不会被保存
def save_grad(name):
    def hook(grad):
        grads[name] = grad

    return hook


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
        #filter boxes which score>thr
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

# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved
from torch.nn.parallel import DataParallel, DistributedDataParallel
import torch.nn as nn


class EnsembleTSModel(nn.Module):
    def __init__(self, modelTeacher, modelStudent):
        super(EnsembleTSModel, self).__init__()

        if isinstance(modelTeacher, (DistributedDataParallel, DataParallel)):
            modelTeacher_ds = modelTeacher.module
        # if isinstance(modelTeacher_dp, (DistributedDataParallel, DataParallel)):
        #     modelTeacher_dp = modelTeacher_dp.module
        if isinstance(modelStudent, (DistributedDataParallel, DataParallel)):
            modelStudent = modelStudent.module
        # if isinstance(modelDomain, (DistributedDataParallel, DataParallel)):
        #     modelStudent = modelDomain.module

        self.modelTeacher = modelTeacher
        # self.modelTeacher_dp = modelTeacher_dp
        self.modelStudent = modelStudent
        # self.modelDomain = modelDomain

# Adaptive Teacher Trainer
class ATeacherTrainer(DefaultTrainer):
    def __init__(self, cfg):
        """
        Args:
            cfg (CfgNode):
        Use the custom checkpointer, which loads other backbone models
        with matching heuristics.
        """
        cfg = DefaultTrainer.auto_scale_workers(cfg, comm.get_world_size())
        data_loader = self.build_train_loader(cfg)

        # create an student model
        model = self.build_model(cfg)
        optimizer = self.build_optimizer(cfg, model)

        # create an teacher model
        model_teacher = self.build_model(cfg)
        self.model_teacher = model_teacher

        # model_teacher = self.build_model(cfg)
        # self.model_teacher_dp = model_teacher



        # model_domain = self.build_model(cfg)
        # self.model_domain = model_domain

        # For training, wrap with DDP. But don't need this for inference.
        if comm.get_world_size() > 1:
            model = DistributedDataParallel(
                model, device_ids=[comm.get_local_rank()], broadcast_buffers=False
            )

        TrainerBase.__init__(self)
        self._trainer = (AMPTrainer if cfg.SOLVER.AMP.ENABLED else SimpleTrainer)(
            model, data_loader, optimizer
        )
        self.scheduler = self.build_lr_scheduler(cfg, optimizer)

        # Ensemble teacher and student model is for model saving and loading
        ensem_ts_model = EnsembleTSModel(model_teacher, model)

        self.checkpointer = DetectionTSCheckpointer(
            ensem_ts_model,
            cfg.OUTPUT_DIR,
            optimizer=optimizer,
            scheduler=self.scheduler,
        )
        self.start_iter = 0
        self.start_round = 0
        self.max_iter = cfg.SOLVER.MAX_ITER
        self.max_round = cfg.SOLVER.MAX_ROUND
        self.cfg = cfg

        self.probe = OpenMatchTrainerProbe(cfg)
        self.register_hooks(self.build_hooks())

    def resume_or_load(self, resume=True):
        """
        If `resume==True` and `cfg.OUTPUT_DIR` contains the last checkpoint (defined by
        a `last_checkpoint` file), resume from the file. Resuming means loading all
        available states (eg. optimizer and scheduler) and update iteration counter
        from the checkpoint. ``cfg.MODEL.WEIGHTS`` will not be used.
        Otherwise, this is considered as an independent training. The method will load model
        weights from the file `cfg.MODEL.WEIGHTS` (but will not load other states) and start
        from iteration 0.
        Args:
            resume (bool): whether to do resume or not
        """
        checkpoint = self.checkpointer.resume_or_load(
            self.cfg.MODEL.WEIGHTS, resume=resume
        )
        # checkpoint_dp = self.checkpointer.resume_or_load(
        #     self.cfg.MODEL.WEIGHTS_DP, resume=resume
        # )
        if resume and self.checkpointer.has_checkpoint():
            self.start_iter = checkpoint.get("iteration", -1) + 1
            # self.start_iter = checkpoint_dp.get("iteration", -1) + 1
            # The checkpoint stores the training iteration that just finished, thus we start
            # at the next iteration (or iter zero if there's no checkpoint).
        if isinstance(self.model, DistributedDataParallel):
            # broadcast loaded data/model from the first rank, because other
            # machines may not have access to the checkpoint file
            # if TORCH_VERSION >= (1, 7):
            #     self.model._sync_params_and_buffers()
            self.start_iter = comm.all_gather(self.start_iter)[0]

    @classmethod
    def build_evaluator(cls, cfg, dataset_name, output_folder=None):
        if output_folder is None:
            output_folder = os.path.join(cfg.OUTPUT_DIR, "inference")
        evaluator_list = []
        evaluator_type = MetadataCatalog.get(dataset_name).evaluator_type

        if evaluator_type == "coco":
            evaluator_list.append(COCOEvaluator(
                dataset_name, output_dir=output_folder))
        elif evaluator_type == "pascal_voc":
            return PascalVOCDetectionEvaluator(dataset_name)
        elif evaluator_type == "pascal_voc_water":
            return PascalVOCDetectionEvaluator(dataset_name,
                                               target_classnames=["bicycle", "bird", "car", "cat", "dog", "person"])
        if len(evaluator_list) == 0:
            raise NotImplementedError(
                "no Evaluator for the dataset {} with the type {}".format(
                    dataset_name, evaluator_type
                )
            )
        elif len(evaluator_list) == 1:
            return evaluator_list[0]

        return DatasetEvaluators(evaluator_list)

    @classmethod
    def build_train_loader(cls, cfg):
        mapper = DatasetMapperTwoCropSeparate(cfg, True)
        return build_detection_semisup_train_loader_two_crops(cfg, mapper)

    @classmethod
    def build_lr_scheduler(cls, cfg, optimizer):
        return build_lr_scheduler(cfg, optimizer)

    def train(self,teacher_ds,teacher_dp):
        self.train_loop(self.start_iter, self.max_iter,self.start_round,self.max_round,teacher_ds,teacher_dp)
        if hasattr(self, "_last_eval_results") and comm.is_main_process():
            verify_results(self.cfg, self._last_eval_results)
            return self._last_eval_results

    def train_loop(self, start_iter: int, max_iter: int,start_round: int, max_round: int,teachers_ds,teacher_dp):
        logger = logging.getLogger(__name__)
        logger.info("Starting training from iteration {}".format(start_iter))

        self.iter = self.start_iter = start_iter
        self.max_iter = max_iter
        self.round = self.start_round = start_round
        self.max_round = max_round

        with EventStorage(start_iter) as self.storage:
            try:
                self.before_train()
                # recall_list = []
                # precise_list = []
                # mIOU_list = []
                # t_ins_score_list = []
                # t_score_list = []
                # s_ins_score_list = []
                # s_score_list = []
                # level_diction_list = dict(p2=[], p3=[], p4=[], p5=[], p6=[])
                # source_level_diction_list = dict(p2=[], p3=[], p4=[], p5=[], p6=[])
                for self.round in range(start_round,max_round):
                    for self.iter in range(start_iter, max_iter):
                    # for self.iter in range(2000, max_iter):
                        self.before_step()
                        self.run_step_full_semisup(teachers_ds,teacher_dp)
                        self.after_step()
                    start_iter = 0


            except Exception:
                logger.exception("Exception during training:")
                raise
            finally:
                self.after_train()

    def NonMaxSuppression(self, proposal_bbox_inst, confi_thres=0.7,nms_thresh = 0.45, proposal_type="roih"):
        if proposal_type == "roih":
            valid_map = proposal_bbox_inst.scores > confi_thres

            # create instances containing boxes and gt_classes
            image_shape = proposal_bbox_inst.image_size
            new_proposal_inst = Instances(image_shape)

            # create box  #actually no need valid_map
            new_bbox_loc = proposal_bbox_inst.pred_boxes.tensor[valid_map, :]
            new_score = proposal_bbox_inst.scores[valid_map,:]
            new_class = proposal_bbox_inst.pred_classes[valid_map,:]
            # new_bbox_loc = proposal_bbox_inst.pred_boxes.tensor
            # new_score = proposal_bbox_inst.scores
            # new_class = proposal_bbox_inst.pred_classes
            scores,index = new_score.sort(descending = True)
            keep_inds = []
            while(len(index) > 0):
                cur_inx = index[0]
                cur_score = scores[cur_inx]
                if cur_score < confi_thres:
                    break;
                keep = True
                for ind in keep_inds:
                    current_bbox = new_bbox_loc[cur_inx]
                    remain_box = new_bbox_loc[ind]
                    # iou = 1
                    ioc = self.box_ioc_xyxy(current_bbox,remain_box)
                    if ioc > nms_thresh:
                        keep = False
                        break

                if keep:
                    keep_inds.append(cur_inx)
                index = index[1:]
            # if len(keep_inds) == 0:
            #     valid_map = proposal_bbox_inst.scores > thres
            #
            #     # create instances containing boxes and gt_classes
            #     image_shape = proposal_bbox_inst.image_size
            #     new_proposal_inst = Instances(image_shape)
            #
            #     # create box
            #     new_bbox_loc = proposal_bbox_inst.pred_boxes.tensor[valid_map, :]
            #     new_boxes = Boxes(new_bbox_loc)
            #
            #     # add boxes to instances
            #     new_proposal_inst.gt_boxes = new_boxes
            #     new_proposal_inst.gt_classes = proposal_bbox_inst.pred_classes[valid_map]
            #     new_proposal_inst.scores = proposal_bbox_inst.scores[valid_map]
            #     for i in new_proposal_inst.scores:
            #         i = 0
            #     return new_proposal_inst



            keep_inds = torch.tensor(keep_inds)
            score_nms = new_score[keep_inds.long()]
            # score_nms = score_nms.reshape(-1,1)
            # score_nms = score_nms.reshape(-1)
            box_nms = new_bbox_loc[keep_inds.long()]
            box_nms = box_nms.reshape(-1,4)
            box_nms = Boxes(box_nms)
            class_nms = new_class[keep_inds.long()]
            # class_nms = class_nms.reshape(-1,1)
            new_proposal_inst.gt_boxes = box_nms
            new_proposal_inst.gt_classes = class_nms
            new_proposal_inst.scores = score_nms


        elif proposal_type == "rpn":

            raise ValueError("Unknown NMS branches")

        return new_proposal_inst
    # =====================================================
    # ================== Pseduo-labeling ==================
    # =====================================================
    def threshold_bbox(self, proposal_bbox_inst, thres=0.7, proposal_type="roih"):
        if proposal_type == "rpn":
            valid_map = proposal_bbox_inst.objectness_logits > thres

            # create instances containing boxes and gt_classes
            image_shape = proposal_bbox_inst.image_size
            new_proposal_inst = Instances(image_shape)

            # create box
            new_bbox_loc = proposal_bbox_inst.proposal_boxes.tensor[valid_map, :]
            new_boxes = Boxes(new_bbox_loc)

            # add boxes to instances
            new_proposal_inst.gt_boxes = new_boxes
            new_proposal_inst.objectness_logits = proposal_bbox_inst.objectness_logits[
                valid_map
            ]
        elif proposal_type == "roih":
            valid_map = proposal_bbox_inst.scores > thres

            # create instances containing boxes and gt_classes
            image_shape = proposal_bbox_inst.image_size
            new_proposal_inst = Instances(image_shape)

            # create box
            new_bbox_loc = proposal_bbox_inst.pred_boxes.tensor[valid_map, :]
            new_boxes = Boxes(new_bbox_loc)

            # add boxes to instances
            new_proposal_inst.gt_boxes = new_boxes
            new_proposal_inst.gt_classes = proposal_bbox_inst.pred_classes[valid_map]
            new_proposal_inst.scores = proposal_bbox_inst.scores[valid_map]

        return new_proposal_inst

    def box_ioc_xyxy(self,box1_rank,box2_sus):

        xA = max(box1_rank[0],box2_sus[0])
        yA = max(box1_rank[1],box2_sus[1])
        xB = min(box1_rank[2], box2_sus[2])
        yB = min(box1_rank[3], box2_sus[3])

        intersect = max(0,xB - xA + 1) * max(0,yB - yA + 1)
        # box_area1 = (box1[2]-box1[0] + 1) * (box1[3] - box1[1] + 1)
        box_area2 = (box2_sus[2] - box2_sus[0] + 1) * (box2_sus[3] - box2_sus[1] + 1)

        # ioc = intersect / float(box_area2 + box_area1 -intersect)
        ioc = intersect / float(box_area2)
        return ioc

    def Knowlegde_Fusion(self,proposals_T, proposals_S, iou_thr=0.5, skip_box_thr=0.05, weights=[1, 1]):
        assert len(proposals_T) == len(proposals_S)
        list_instances = []
        num_proposal_output = 0.0
        for i in range(len(proposals_T)):
            pseudo_label_inst = self.pseudo_fusion(proposals_T[i], proposals_S[i], iou_thr, skip_box_thr, weights)

            num_proposal_output += len(pseudo_label_inst)
            list_instances.append(pseudo_label_inst)
        num_proposal_output = num_proposal_output / (len(proposals_T) + len(proposals_S))
        return list_instances, num_proposal_output

    def pseudo_fusion(self,output_t, output_s, iou_thr=0.5, skip_box_thr=0.05, weights=[1, 1]):

        image_size = output_t.image_size

        boxes_list, scores_list, labels_list = [], [], []

        box_list_t = output_t.pred_boxes.tensor
        scores_list_t = output_t.scores
        classes_list_t = output_t.pred_classes

        box_list_s = output_s.pred_boxes.tensor
        scores_list_s = output_s.scores
        classes_list_s = output_s.pred_classes

        boxes_list.append(box_list_t)
        boxes_list.append(box_list_s)
        scores_list.append(scores_list_t)
        scores_list.append(scores_list_s)
        labels_list.append(classes_list_t)
        labels_list.append(classes_list_s)
        boxes, scores, labels = weighted_boxes_fusion(boxes_list, scores_list, labels_list, weights=weights,
                                                      iou_thr=iou_thr, skip_box_thr=skip_box_thr)
        result = Instances(image_size)
        boxes = Boxes(torch.tensor(boxes).cuda())
        boxes.clip(image_size)
        result.gt_boxes = boxes
        result.scores = torch.tensor(scores).cuda()
        result.gt_classes = torch.tensor(labels).cuda().long()
        return result

    def NonMaxSuppression(self, proposal_bbox_inst, confi_thres=0.9,nms_thresh = 0.99, proposal_type="roih"):
        if proposal_type == "roih":
            valid_map = proposal_bbox_inst.scores > confi_thres

            # create instances containing boxes and gt_classes
            image_shape = proposal_bbox_inst.image_size
            new_proposal_inst = Instances(image_shape)

            # create box  #actually no need valid_map
            new_bbox_loc = proposal_bbox_inst.pred_boxes.tensor[valid_map, :]
            new_score = proposal_bbox_inst.scores[valid_map]
            new_class = proposal_bbox_inst.pred_classes[valid_map]

            # new_bbox_loc = proposal_bbox_inst.pred_boxes.tensor
            # new_score = proposal_bbox_inst.scores
            # new_class = proposal_bbox_inst.pred_classes
            scores,index = new_score.sort(descending = True)
            keep_inds = []
            while(len(index) > 0):
                cur_inx = index[0]
                cur_score = scores[cur_inx]
                if cur_score < confi_thres:
                    index = index[1:]
                    continue
                keep = True
                for ind in keep_inds:
                    current_bbox = new_bbox_loc[cur_inx]
                    remain_box = new_bbox_loc[ind]
                    # iou = 1
                    ioc = self.box_ioc_xyxy(current_bbox,remain_box)
                    if ioc > nms_thresh:
                        keep = False
                        break
                if keep:
                    keep_inds.append(cur_inx)
                index = index[1:]
            # if len(keep_inds) == 0:
            #     valid_map = proposal_bbox_inst.scores > thres
            #
            #     # create instances containing boxes and gt_classes
            #     image_shape = proposal_bbox_inst.image_size
            #     new_proposal_inst = Instances(image_shape)
            #
            #     # create box
            #     new_bbox_loc = proposal_bbox_inst.pred_boxes.tensor[valid_map, :]
            #     new_boxes = Boxes(new_bbox_loc)
            #
            #     # add boxes to instances
            #     new_proposal_inst.gt_boxes = new_boxes
            #     new_proposal_inst.gt_classes = proposal_bbox_inst.pred_classes[valid_map]
            #     new_proposal_inst.scores = proposal_bbox_inst.scores[valid_map]
            #     for i in new_proposal_inst.scores:
            #         i = 0
            #     return new_proposal_inst

            keep_inds = torch.tensor(keep_inds)
            score_nms = new_score[keep_inds.long()]
            # score_nms = score_nms.reshape(-1,1)
            # score_nms = score_nms.reshape(-1)
            box_nms = new_bbox_loc[keep_inds.long()]
            box_nms = box_nms.reshape(-1,4)
            box_nms = Boxes(box_nms)
            class_nms = new_class[keep_inds.long()]
            # class_nms = class_nms.reshape(-1,1)
            new_proposal_inst.gt_boxes = box_nms
            new_proposal_inst.gt_classes = class_nms
            new_proposal_inst.scores = score_nms

        elif proposal_type == "rpn":

            raise ValueError("Unknown NMS branches")

        return new_proposal_inst

    def process_pseudo_label(
            self, proposals_rpn_unsup_k, cur_threshold, proposal_type, psedo_label_method=""
    ):
        list_instances = []
        num_proposal_output = 0.0
        for proposal_bbox_inst in proposals_rpn_unsup_k:
            # thresholding
            if psedo_label_method == "thresholding":
                proposal_bbox_inst = self.threshold_bbox(
                    proposal_bbox_inst, thres=cur_threshold, proposal_type=proposal_type
                )
            elif psedo_label_method == "NMS":
                proposal_bbox_inst = self.NonMaxSuppression(
                    proposal_bbox_inst, confi_thres=cur_threshold, proposal_type=proposal_type
                )

            else:
                raise ValueError("Unkown pseudo label boxes methods")
            num_proposal_output += len(proposal_bbox_inst)
            list_instances.append(proposal_bbox_inst)
        num_proposal_output = num_proposal_output / len(proposals_rpn_unsup_k)
        return list_instances, num_proposal_output

    def remove_label(self, label_data):
        for label_datum in label_data:
            if "instances" in label_datum.keys():
                del label_datum["instances"]
        return label_data

    def add_label(self, unlabled_data, label):
        for unlabel_datum, lab_inst in zip(unlabled_data, label):
            unlabel_datum["instances"] = lab_inst
        return unlabled_data

    def get_label(self, label_data):
        label_list = []
        for label_datum in label_data:
            if "instances" in label_datum.keys():
                label_list.append(copy.deepcopy(label_datum["instances"]))

        return label_list
    # def consistency_compare(self,pesudo_proposals_roih_unsup_k, gt_proposal, cur_compare_threshold):
    #     consistency = 0.5
    #
    #
    #     return consistency

    # def consistency_compare(self, roi_preds, gt_label, threshold):


    # def get_label_test(self, label_data):
    #     label_list = []
    #     for label_datum in label_data:
    #         if "instances" in label_datum.keys():
    #             label_list.append(label_datum["instances"])

    # =====================================================
    # =================== Training Flow ===================
    # =====================================================


    # def MixupDetection(self,img1,img2,label1,label2,lambd):
    #     # mixup two images
    #     height = max(img1.shape[0], img2.shape[0])
    #     width = max(img1.shape[1], img2.shape[1])
    #     mix_img = mx.nd.zeros(shape=(height, width, 3), dtype='float32')
    #     mix_img[:img1.shape[0], :img1.shape[1], :] = img1.astype('float32') * lambd
    #     mix_img[:img2.shape[0], :img2.shape[1], :] += img2.astype('float32') * (1. - lambd)
    #     mix_img = mix_img.astype('uint8')
    #     y1 = np.hstack((label1, np.full((label1.shape[0], 1), lambd)))
    #     y2 = np.hstack((label2, np.full((label2.shape[0], 1), 1. - lambd)))
    #     mix_label = np.vstack((y1, y2))
    #     return mix_img, mix_label

    def run_step_full_semisup(self,teacher_ds,teacher_dp,):
        self._trainer.iter = self.iter
        assert self.model.training, "[UBTeacherTrainer] model was changed to eval mode!"
        start = time.perf_counter()
        data = next(self._trainer._data_loader_iter)
        # data_q and data_k from different augmentations (q:strong, k:weak)
        # label_strong, label_weak, unlabed_strong, unlabled_weak
        label_train_data_q, label_train_data_k, label_compare_data_q, label_compare_data_k, unlabel_data_q, unlabel_data_k = data
        data_time = time.perf_counter() - start
        name = 520
        # pretained stage
        # firstly,copy a teacher net ,update teacher model per iter
        # train random initialed model(T/S) 4000 iter to get a completely result
        # compared to teacher ds and teacher dp
        if self.iter < self.cfg.SEMISUPNET.INITIAL_ITER:
            if self.iter == self.cfg.SEMISUPNET.BURN_UP_STEP:
                    # update copy the the whole model
                    self._update_teacher_model(keep_rate=0.00)
            #         # self.model.build_discriminator()
            #
            elif (self.iter - self.cfg.SEMISUPNET.BURN_UP_STEP
                ) % self.cfg.SEMISUPNET.TEACHER_UPDATE_ITER == 0:
                    self._update_teacher_model(
                        keep_rate=self.cfg.SEMISUPNET.EMA_KEEP_RATE)
            # if self.iter == (self.max_iter-1):
            #     self._update_teacher_ds_model(teacher_ds,
            #         keep_rate=self.cfg.SEMISUPNET.EMA_KEEP_RATE)
            #     self._update_teacher_dp_model(teacher_dp,
            #         keep_rate=self.cfg.SEMISUPNET.EMA_KEEP_RATE)

        # combination training
        # firstly, use teacher as new student model
        # for each iter
        # uodate teacher every iter
        # update teacher ds/dp every iteration

            record_dict = {}

            ######################## For probe #################################
            # import pdb; pdb. set_trace()
            gt_unlabel_k = self.get_label(unlabel_data_k)
            # gt_unlabel_q = self.get_label_test(unlabel_data_q)

            #  0. remove unlabeled data labels
            unlabel_data_q = self.remove_label(unlabel_data_q)
            unlabel_data_k = self.remove_label(unlabel_data_k)
            # fake_src_data_q = self.remove_label(label_compare_data_q)
            # fake_src_data_k = self.remove_label(label_compare_data_k)

            #  1. generate the pseudo-label using teacher model
            # with torch.no_grad():
            #     (
            #         _,
            #         proposals_rpn_unsup_k,
            #         proposals_roih_unsup_k,
            #         _,
            #     ) = self.model_teacher(unlabel_data_k, branch="unsup_data_weak")
            with torch.no_grad():
                (
                    _,
                    proposals_rpn_unsup_k_s,
                    proposals_roih_unsup_k_s,
                    _,
                ) = teacher_ds(unlabel_data_k, branch="unsup_data_weak")
                (
                    _,
                    proposals_rpn_unsup_k_p,
                    proposals_roih_unsup_k_p,
                    _,
                ) = teacher_dp(unlabel_data_k, branch="unsup_data_weak")
            #
            # # todo:pseudo label fusion
            pesudo_proposals_roih_unsup_k,_ = self.Knowlegde_Fusion(
                proposals_roih_unsup_k_s,proposals_roih_unsup_k_p, self.cfg.SEMISUPNET.FUSION_IOU_THR,self.cfg.SEMISUPNET.FUSION_BBOX_THRESHOLD,self.cfg.SEMISUPNET.FUSION_WEIGHT
            )
            joint_proposal_dict = {}
            joint_proposal_dict["proposals_pseudo_roi_P"] = proposals_roih_unsup_k_s
            joint_proposal_dict["proposals_pseudo_roi_S"] = proposals_roih_unsup_k_p
            ######################## For probe #################################
            # import pdb; pdb. set_trace()

            # probe_metrics = ['compute_fp_gtoutlier', 'compute_num_box']
            # probe_metrics = ['compute_num_box']
            # analysis_pred, _ = self.probe.compute_num_box(gt_unlabel_k,proposals_roih_unsup_k,'pred')
            # record_dict.update(analysis_pred)
            ######################## For probe END #################################

            #  2. Pseudo-labeling
            cur_threshold = self.cfg.SEMISUPNET.BBOX_THRESHOLD

            # unlabel_data_q_P = self.add_label(
            #     unlabel_data_q, joint_proposal_dict["proposals_pseudo_roi_P"]
            # )
            # unlabel_data_q_S = self.add_label(
            #     unlabel_data_q, joint_proposal_dict["proposals_pseudo_roi_S"]
            # )
            joint_proposal_dict["proposals_rpn"] = proposals_rpn_unsup_k_s
            # Process pseudo labels and thresholding
            (
                pesudo_proposals_rpn_unsup_k,
                nun_pseudo_bbox_rpn,
            ) = self.process_pseudo_label(
                proposals_rpn_unsup_k_s, cur_threshold, "rpn", "thresholding"
            )
            # analysis_pred, _ = self.probe.compute_num_box(gt_unlabel_k,pesudo_proposals_rpn_unsup_k,'pred',True)
            # record_dict.update(analysis_pred)

            joint_proposal_dict["proposals_pseudo_rpn"] = pesudo_proposals_rpn_unsup_k
            # Pseudo_labeling for ROI head (bbox location/objectness)
            #todo:123456
            # pesudo_proposals_roih_unsup_k, _ = self.process_pseudo_label(
            #     proposals_roih_unsup_k, cur_threshold, "roih", "thresholding"
            # )

            joint_proposal_dict["proposals_pseudo_roih"] = pesudo_proposals_roih_unsup_k
            # ins_proposal_dict = {}
            joint_proposal_dict["proposals_pseudo"] = proposals_roih_unsup_k_s

            #todo:quality eval
            # for i in range(len(pesudo_proposals_roih_unsup_k)):
            #     temp = pesudo_proposals_roih_unsup_k[i]
            #     # mAP_per_image = get_mAP(temp.gt_boxes,gt_unlabel_k.gt_boxes)
            #     recall_per, preci_per, mIOU_per = get_mIOU(temp, gt_unlabel_k[i])
            # #只有0和1啊，单张图像，感觉。。没什么意义啊
            #     recall_list.append(recall_per)
            #     precise_list.append(preci_per)
            #     mIOU_list.append(mIOU_per)


            # 3. add pseudo-label to unlabeled data

            unlabel_data_q = self.add_label(
                unlabel_data_q, joint_proposal_dict["proposals_pseudo_roih"]
            )
            # unlabel_data_k = self.add_label(
            #     unlabel_data_k, joint_proposal_dict["proposals_pseudo_roih"]
            # )
            unlabel_data_k = self.add_label(
                unlabel_data_k, joint_proposal_dict["proposals_pseudo_roih"]
            )

            all_label_data = label_train_data_q + label_train_data_k

            # 4. input both strongly and weakly augmented labeled data into student model
            record_all_label_data, _, _, _ = self.model(
                all_label_data, branch="supervised"
            )
            record_dict.update(record_all_label_data)

            all_unlabel_data = unlabel_data_q
            # unlabel_data_q_S = self.add_label(
            #     unlabel_data_q, joint_proposal_dict["proposals_pseudo_roih_S"]
            # )
            # unlabel_data_q_P = self.add_label(
            #     unlabel_data_q, joint_proposal_dict["proposals_pseudo_roih_P"]
            # )

            psu_sum = 0
            pseudo_data = []
            for i in range(len(unlabel_data_q)):
                data = unlabel_data_q[i]
                # print(len(data['instances']))
                if len(data['instances'])!=0:

                    pseudo_data.append(data)
                else:
                    # pseudo_data.append(label_compare_data_k[i])
                    psu_sum +=1

            # record_all_unlabel_data_P, _, _, _ = self.model(
            #     unlabel_data_q_P, branch="supervised_target"
            # )
            #
            # record_all_unlabel_data_S, _, _, _ = self.model(
            #     unlabel_data_q_S, branch="supervised_target"
            # )



            record_all_unlabel_data, _, _, _ = self.model(
                all_unlabel_data, branch="supervised_target"
            )
            new_record_all_unlabel_data = {}

            for key in record_all_unlabel_data.keys():
                new_record_all_unlabel_data[key + "_pseudo"] = record_all_unlabel_data[
                    key
                ]
            record_dict.update(new_record_all_unlabel_data)

            # new_record_all_unlabel_data_S = {}
            # for key in record_all_unlabel_data_S.keys():
            #     new_record_all_unlabel_data_S[key + "_pseudo_S"] = record_all_unlabel_data_S[
            #         key
            #     ]
            # record_dict.update(new_record_all_unlabel_data_S)
            #
            # new_record_all_unlabel_data_P = {}
            # for key in record_all_unlabel_data_P.keys():
            #     new_record_all_unlabel_data_P[key + "_pseudo_P"] = record_all_unlabel_data_P[
            #         key
            #     ]
            # record_dict.update(new_record_all_unlabel_data_P)

            # 6. input weakly labeled data (source) and weakly unlabeled data (target) to student model
            # give sign to the target data
            # unlabel_data_k =

            for i_index in range(len(unlabel_data_k)):
                # unlabel_data_item = {}
                for k, v in unlabel_data_k[i_index].items():
                    # label_data_k[i_index][k + "_unlabeled"] = v
                    label_train_data_k[i_index][k + "_unlabeled"] = v
                # unlabel_data_k[i_index] = unlabel_data_item

            all_domain_data = label_train_data_k
            # all_domain_data = label_data_k + unlabel_data_k
            # record_all_domain_data = self.model(all_domain_data, branch="domain")
            # record_dict.update(record_all_domain_data)
            # for i in range(len(unlabel_data_k)):
            #     t_ins_list.append(t_ins_score[i])
            #     t_list.append(t_img_score[i])
            #     s_ins_list.append(s_ins_score[i])
            #     s_list.append(s_img_score[i])
            #     for level in level_diction_list:
            #         level_diction_list[level].append(level_diction[level][i])
            #         source_level_diction_list[level].append(source_level_diction[level][i])

            # weight losses
            loss_dict = {}
            for key in record_dict.keys():
                if key.startswith("loss"):
                    if key == "loss_rpn_loc_pseudo" or key == "loss_box_reg_pseudo":
                        # pseudo bbox regression <- 0
                        loss_dict[key] = record_dict[key] * 0
                    elif key[-6:] == "pseudo":  # unsupervised loss
                        loss_dict[key] = (
                                record_dict[key] *
                                self.cfg.SEMISUPNET.UNSUP_LOSS_WEIGHT
                        )
                    elif (
                            key == "loss_D_img_s" or key == "loss_D_img_t"
                    ):  # set weight for discriminator
                        # import pdb
                        # pdb.set_trace()
                        loss_dict[key] = record_dict[
                                             key] * 0  # Need to modify defaults and yaml
                    elif (key == "loss_D_ins_t" or key == "loss_D_ins_s"):
                        loss_dict[key] = record_dict[
                                             key] * 0

                    else:  # supervised loss
                        loss_dict[key] = record_dict[key] * 1


        else:
            if self.iter == self.cfg.SEMISUPNET.INITIAL_ITER:
                    # update copy the the whole model
                    self._update_student_model(keep_rate=0.00)
            if (self.iter - self.cfg.SEMISUPNET.INITIAL_ITER
            ) % self.cfg.SEMISUPNET.UPDATE_ITER == 0:
                # self._update_teacher_model(
                #     keep_rate=self.cfg.SEMISUPNET.EMA_KEEP_RATE)
                self._update_teacher_ds_model(teacher_ds,
                                              keep_rate=self.cfg.SEMISUPNET.EMA_KEEP_RATE)
                self._update_teacher_dp_model(teacher_dp,
                                              keep_rate=self.cfg.SEMISUPNET.EMA_KEEP_RATE)

            record_dict = {}

            ######################## For probe #################################
            # import pdb; pdb. set_trace()
            gt_unlabel_k = self.get_label(unlabel_data_k)
            # gt_unlabel_q = self.get_label_test(unlabel_data_q)

            #  0. remove unlabeled data labels
            unlabel_data_q = self.remove_label(unlabel_data_q)
            unlabel_data_k = self.remove_label(unlabel_data_k)
            # fake_src_data_q = self.remove_label(label_compare_data_q)
            # fake_src_data_k = self.remove_label(label_compare_data_k)

            #  1. generate the pseudo-label using teacher model
            # with torch.no_grad():
            #     (
            #         _,
            #         proposals_rpn_unsup_k,
            #         proposals_roih_unsup_k,
            #         _,
            #     ) = self.model_teacher(unlabel_data_k, branch="unsup_data_weak")
            with torch.no_grad():
                (
                    _,
                    proposals_rpn_unsup_k,
                    proposals_roih_unsup_k,
                    _,
                ) = teacher_ds(unlabel_data_k, branch="unsup_data_weak")
                (
                    _,
                    proposals_rpn_unsup_k_p,
                    proposals_roih_unsup_k_p,
                    _,
                ) = teacher_dp(unlabel_data_k, branch="unsup_data_weak")
            #
            # # todo:pseudo label fusion
            pesudo_proposals_roih_unsup_k, _ = self.Knowlegde_Fusion(
                proposals_roih_unsup_k, proposals_roih_unsup_k_p, self.cfg.SEMISUPNET.FUSION_IOU_THR,
                self.cfg.SEMISUPNET.FUSION_BBOX_THRESHOLD, self.cfg.SEMISUPNET.FUSION_WEIGHT
            )

            #  2. Pseudo-labeling
            cur_threshold = self.cfg.SEMISUPNET.BBOX_THRESHOLD

            joint_proposal_dict = {}
            joint_proposal_dict["proposals_rpn"] = proposals_rpn_unsup_k
            # Process pseudo labels and thresholding
            (
                pesudo_proposals_rpn_unsup_k,
                nun_pseudo_bbox_rpn,
            ) = self.process_pseudo_label(
                proposals_rpn_unsup_k, cur_threshold, "rpn", "thresholding"
            )
            # analysis_pred, _ = self.probe.compute_num_box(gt_unlabel_k,pesudo_proposals_rpn_unsup_k,'pred',True)
            # record_dict.update(analysis_pred)

            joint_proposal_dict["proposals_pseudo_rpn"] = pesudo_proposals_rpn_unsup_k
            # Pseudo_labeling for ROI head (bbox location/objectness)
            # todo:123456
            # pesudo_proposals_roih_unsup_k, _ = self.process_pseudo_label(
            #     proposals_roih_unsup_k, cur_threshold, "roih", "thresholding"
            # )

            joint_proposal_dict["proposals_pseudo_roih"] = pesudo_proposals_roih_unsup_k
            # ins_proposal_dict = {}
            joint_proposal_dict["proposals_pseudo"] = proposals_roih_unsup_k

            # todo:quality eval
            # for i in range(len(pesudo_proposals_roih_unsup_k)):
            #     temp = pesudo_proposals_roih_unsup_k[i]
            #     # mAP_per_image = get_mAP(temp.gt_boxes,gt_unlabel_k.gt_boxes)
            #     recall_per, preci_per, mIOU_per = get_mIOU(temp, gt_unlabel_k[i])
            # #只有0和1啊，单张图像，感觉。。没什么意义啊
            #     recall_list.append(recall_per)
            #     precise_list.append(preci_per)
            #     mIOU_list.append(mIOU_per)

            # 3. add pseudo-label to unlabeled data

            unlabel_data_q = self.add_label(
                unlabel_data_q, joint_proposal_dict["proposals_pseudo_roih"]
            )
            # unlabel_data_k = self.add_label(
            #     unlabel_data_k, joint_proposal_dict["proposals_pseudo_roih"]
            # )
            unlabel_data_k = self.add_label(
                unlabel_data_k, joint_proposal_dict["proposals_pseudo_roih"]
            )

            all_label_data = label_train_data_q + label_train_data_k

            # 4. input both strongly and weakly augmented labeled data into student model
            record_all_label_data, _, _, _ = self.model(
                all_label_data, branch="supervised"
            )
            record_dict.update(record_all_label_data)

            all_unlabel_data = unlabel_data_q
            psu_sum = 0
            pseudo_data = []
            for i in range(len(unlabel_data_q)):
                data = unlabel_data_q[i]
                # print(len(data['instances']))
                if len(data['instances']) != 0:

                    pseudo_data.append(data)
                else:
                    # pseudo_data.append(label_compare_data_k[i])
                    psu_sum += 1

            record_all_unlabel_data, _, _, _ = self.model(
                all_unlabel_data, branch="supervised_target"
            )
            # record_all_unlabel_data, _, _, _ = self.model(
            #     all_unlabel_data, branch="supervised_target"
            # )

            new_record_all_unlabel_data = {}
            for key in record_all_unlabel_data.keys():
                new_record_all_unlabel_data[key + "_pseudo"] = record_all_unlabel_data[
                    key
                ]
            record_dict.update(new_record_all_unlabel_data)

            # 6. input weakly labeled data (source) and weakly unlabeled data (target) to student model
            # give sign to the target data
            # unlabel_data_k =

            for i_index in range(len(unlabel_data_k)):
                # unlabel_data_item = {}
                for k, v in unlabel_data_k[i_index].items():
                    # label_data_k[i_index][k + "_unlabeled"] = v
                    label_train_data_k[i_index][k + "_unlabeled"] = v
                # unlabel_data_k[i_index] = unlabel_data_item

            all_domain_data = label_train_data_k
            # all_domain_data = label_data_k + unlabel_data_k
            # record_all_domain_data = self.model(all_domain_data, branch="domain")
            # record_dict.update(record_all_domain_data)
            # for i in range(len(unlabel_data_k)):
            #     t_ins_list.append(t_ins_score[i])
            #     t_list.append(t_img_score[i])
            #     s_ins_list.append(s_ins_score[i])
            #     s_list.append(s_img_score[i])
            #     for level in level_diction_list:
            #         level_diction_list[level].append(level_diction[level][i])
            #         source_level_diction_list[level].append(source_level_diction[level][i])

            # weight losses
            loss_dict = {}
            for key in record_dict.keys():
                if key.startswith("loss"):
                    if key == "loss_rpn_loc_pseudo" or key == "loss_box_reg_pseudo":
                        # pseudo bbox regression <- 0
                        loss_dict[key] = record_dict[key] * 0
                    elif key[-6:] == "pseudo":  # unsupervised loss
                        loss_dict[key] = (
                                record_dict[key] *
                                self.cfg.SEMISUPNET.UNSUP_LOSS_WEIGHT
                        )
                    elif (
                            key == "loss_D_img_s" or key == "loss_D_img_t"
                    ):  # set weight for discriminator
                        # import pdb
                        # pdb.set_trace()
                        loss_dict[key] = record_dict[
                                             key] * 0  # Need to modify defaults and yaml
                    elif (key == "loss_D_ins_t" or key == "loss_D_ins_s"):
                        loss_dict[key] = record_dict[
                                             key] * 0

                    else:  # supervised loss
                        loss_dict[key] = record_dict[key] * 1




        losses = sum(loss_dict.values())

        metrics_dict = loss_dict
        metrics_dict["name"] = name
        metrics_dict["data_time"] = data_time

        self._write_metrics(metrics_dict)

        self.optimizer.zero_grad()
        losses.backward()
        self.optimizer.step()


#         if len(t_ins_list)%800 == 0:
#             assert len(t_list) == len(t_ins_list) == len(mIOU_list) == len(recall_list) == len(precise_list)
#
#             #target img_dis scores for recall/precise
#             t_dis = []
#             t_ins_dis = []
#             bar_recall_t = collections.Counter(t_dis)
#             bar_precis_t = collections.Counter(t_dis)
#
#             bar_recall_t_ins = collections.Counter(t_ins_dis)
#             bar_precis_t_ins = collections.Counter(t_ins_dis)
#             # bar_num = len(x_train)
#             # for i in range(0, 11):
#             #     bar_recall_t[i] = 0
#             #     bar_precis_t[i] = 0
#             for ele in t_list:
#                 t_dis.append(int(np.round(ele * 10)))
#
#             for ele in t_ins_list:
#                 t_ins_dis.append(int(np.round(ele * 10)))
#
#             len_bar = collections.Counter(t_dis)
#             len_ins_bar = collections.Counter(t_ins_dis)
#             for i in range(len(t_dis)):
#                 bar_recall_t[t_dis[i]] += recall_list[i]
#                 bar_precis_t[t_dis[i]] += precise_list[i]
#
#                 bar_recall_t_ins[t_ins_dis[i]] += recall_list[i]
#                 bar_precis_t_ins[t_ins_dis[i]] += precise_list[i]
#
#             pre_t_img_show = []
#             rec_t_img_show = []
#
#             pre_t_ins_show = []
#             rec_t_ins_show = []
#             for i in range(0,11):
#                 s_p = bar_precis_t[i]/(len_bar[i]+0.001)
#                 s_r = bar_recall_t[i]/(len_bar[i]+0.001)
#
#                 s_ins_p = bar_precis_t_ins[i] / (len_ins_bar[i] + 0.001)
#                 s_ins_r = bar_recall_t_ins[i] / (len_ins_bar[i] + 0.001)
#
#                 pre_t_img_show.append(s_p)
#                 rec_t_img_show.append(s_r)
#
#                 pre_t_ins_show.append(s_ins_p)
#                 rec_t_ins_show.append(s_ins_r)
#
#             #source_list_level_each
#             plt.figure(8)
#             level_show = dict(p2=[], p3=[], p4=[], p5=[], p6=[])
#             list_level_source = dict(p2=[], p3=[], p4=[], p5=[], p6=[])
#             levels = ["p2", "p3", "p4", "p5", "p6"]
#             for level in levels:
#                 for ele in source_level_diction_list[level]:
#                     list_level_source[level].append(int(np.round(ele * 10)))
#             x_axis = np.arange(11).astype(dtype=np.str)
#             for level in levels:
#                 x_train = collections.Counter(list_level_source[level])
#                 for i in range(0, 11):
#                     s = x_train[i]
#                     level_show[level].append(s)
#             plt.subplot(851)
#             plt.bar(x_axis, level_show["p2"], width=0.5)
#             plt.title('source_p2')
#             plt.subplot(852)
#             plt.bar(x_axis, level_show["p3"], width=0.5)
#             plt.title('source_p3')
#             plt.subplot(853)
#             plt.bar(x_axis, level_show["p4"], width=0.5)
#             plt.title('source_p4')
#             plt.subplot(854)
#             plt.bar(x_axis, level_show["p5"], width=0.5)
#             plt.title('source_p5')
#             plt.subplot(855)
#             plt.bar(x_axis, level_show["p6"], width=0.5)
#             plt.title('source_p6')
#             plt.savefig("source_level_img_bar_iter_{}.png".format(len(t_ins_list) / 4))
#
# #target_distribution
#             plt.figure(9)
#             level_show_t = dict(p2=[], p3=[], p4=[], p5=[], p6=[])
#             list_level_target = dict(p2=[], p3=[], p4=[], p5=[], p6=[])
#             for level in levels:
#                 for ele in level_diction_list[level]:
#                     list_level_target[level].append(int(np.round(ele * 10)))
#             for level in levels:
#                 x_train = collections.Counter(list_level_target[level])
#                 for i in range(0, 11):
#                     s = x_train[i]
#                     level_show_t[level].append(s)
#             plt.subplot(951)
#             plt.bar(x_axis, level_show_t["p2"], width=0.5)
#             plt.title('source_p2')
#             plt.subplot(952)
#             plt.bar(x_axis, level_show_t["p3"], width=0.5)
#             plt.title('source_p3')
#             plt.subplot(953)
#             plt.bar(x_axis, level_show_t["p4"], width=0.5)
#             plt.title('source_p4')
#             plt.subplot(954)
#             plt.bar(x_axis, level_show_t["p5"], width=0.5)
#             plt.title('source_p5')
#             plt.subplot(955)
#             plt.bar(x_axis, level_show_t["p6"], width=0.5)
#             plt.title('source_p6')
#             plt.savefig("target_level_img_bar_iter_{}.png".format(len(t_ins_list) / 4))
#
#
#             #img
#             plt.figure(6)
#             plt.subplot(621)
#             plt.bar(x_axis,pre_t_img_show,width=0.5)
#             plt.xlabel('t_img_dis')
#             plt.ylabel('precise_score')
#             plt.title('target_dis_img & precise')
#
#             plt.subplot(622)
#             plt.bar(x_axis, rec_t_img_show, width=0.5)
#             plt.xlabel('t_img_dis')
#             plt.ylabel('recall_score')
#             plt.title('target_dis_img & recall')
#             plt.savefig("prediction_quailty_with_domain_img_bar_iter_{}.png".format(len(t_ins_list) / 4))
#
#             #ins
#             plt.figure(7)
#             plt.subplot(721)
#             plt.bar(x_axis, pre_t_ins_show, width=0.5)
#             plt.xlabel('t_ins_dis')
#             plt.ylabel('precise_score')
#             plt.title('target_dis_ins & precise')
#
#             plt.subplot(722)
#             plt.bar(x_axis, rec_t_ins_show, width=0.5)
#             plt.xlabel('t_ins_dis')
#             plt.ylabel('recall_score')
#             plt.title('target_dis_ins & recall')
#             plt.savefig("prediction_quailty_with_domain_ins_bar_iter_{}.png".format(len(t_ins_list) / 4))
#
#             #scatter for miou and dis_img
#             plt.figure(1)
#
#
#             plt.subplot(133)
#             plt.scatter(mIOU_list, t_list,s = 1)
#             plt.xlabel('mIOU')
#             plt.ylabel('img_dis')
#             plt.title('mIOU & domain_img')
#             plt.savefig("prediction_quailty_with_domain_img_iter_{}.png".format(len(t_ins_list) / 4))
#             #
#
#             #scatter for miou and dis_ins
#             plt.figure(4)
#
#
#             plt.subplot(433)
#             plt.scatter(mIOU_list, t_ins_list,s=1)
#             plt.xlabel('mIOU')
#             plt.ylabel('ins_dis')
#             plt.title('iou & domain_ins')
#             #
#             plt.savefig("prediction_quailty_with_domain_ins_iter_{}.png".format(len(t_ins_list)/4))
#
#
#             #bar of s_ins_num and s_img_num
#
#             plt.figure(5)
#             plt.subplot(531)
#             list_ins_source = []
#             ppt_s_ins = []
#             for ele in s_ins_list:
#                 list_ins_source.append(int(np.round(ele * 10)))
#             x_train = collections.Counter(list_ins_source)
#             x_num = len(x_train)
#             x_axis = np.arange(11).astype(dtype=np.str)
#             for i in range(0, 11):
#                 s = x_train[i]
#                 ppt_s_ins.append(s)
#             plt.bar(x_axis, ppt_s_ins, width=0.5)
#             # plt.savefig('ins_s.png')
#             plt.xlabel('s_ins')
#             plt.ylabel('dis_num')
#             plt.title('source ins')
#
#             plt.subplot(532)
#             list_img_source = []
#             ppt_s_img = []
#             for ele in s_list:
#                 list_img_source.append(int(np.round(ele * 10)))
#             x_train = collections.Counter(list_img_source)
#             x_num = len(x_train)
#             x_axis = np.arange(11).astype(dtype=np.str)
#             for i in range(0, 11):
#                 s = x_train[i]
#                 ppt_s_img.append(s)
#             plt.bar(x_axis, ppt_s_img, width=0.5)
#             # plt.savefig('ins_s.png')
#             plt.xlabel('s_img')
#             plt.ylabel('dis_num')
#             plt.title('source img')
#
#             plt.savefig("source_iter_reiou_{}.png".format(len(t_ins_list) / 4))
#
#
#             #bar of target_ins_num and target_img_num
#             plt.figure(3)
#
#             plt.subplot(331)
#             list_ins_target = []
#             ppt_t_ins = []
#             for ele in t_ins_list:
#                 try:
#                     list_ins_target.append(int(np.round(ele * 10)))
#                 except:
#                     continue
#             x_train = collections.Counter(list_ins_target)
#             x_num = len(x_train)
#             x_axis = np.arange(11).astype(dtype=np.str)
#             for i in range(0, 11):
#                 s = x_train[i]
#                 ppt_t_ins.append(s)
#             plt.bar(x_axis, ppt_t_ins, width=0.5)
#             # plt.savefig('ins_s.png')
#             plt.xlabel('t_ins')
#             plt.ylabel('dis_num')
#             plt.title('target ins')
#
#             plt.subplot(332)
#             list_img_target = []
#             ppt_t_img = []
#             for ele in t_list:
#                 list_img_target.append(int(np.round(ele * 10)))
#             x_train = collections.Counter(list_img_target)
#             x_num = len(x_train)
#             x_axis = np.arange(11).astype(dtype=np.str)
#             for i in range(0, 11):
#                 s = x_train[i]
#                 ppt_t_img.append(s)
#             plt.bar(x_axis, ppt_t_img, width=0.5)
#             # plt.savefig('ins_s.png')
#             plt.xlabel('t_img')
#             plt.ylabel('dis_num')
#             plt.title('target img')
#
#             plt.savefig("target_iter_reiou_{}.png".format(len(t_ins_list) / 4))



            # plt.subplot(331)
            # plt.scatter(s_ins_list, s_list, s=1)
            # plt.xlabel('S_ins')
            # plt.ylabel('S_img')
            # plt.title('source ins & img')

            # plt.subplot(332)
            # plt.scatter(t_ins_list, t_list, s=1)
            # plt.xlabel('t_ins')
            # plt.ylabel('t_img')
            # plt.title('target ins & img')


            # plt.savefig("source_iter_{}.png".format(len(t_ins_list) / 4))

        # return psu_sum

        # self._trainer.iter = self.iter
        # assert self.model.training, "[UBTeacherTrainer] model was changed to eval mode!"
        # start = time.perf_counter()
        # data = next(self._trainer._data_loader_iter)
        # # data_q and data_k from different augmentations (q:strong, k:weak)
        # # label_strong, label_weak, unlabed_strong, unlabled_weak
        # label_data_q, label_data_k,label_compare_data_q,label_compare_data_k, unlabel_data_q, unlabel_data_k = data
        # data_time = time.perf_counter() - start
        #
        # # burn-in stage (supervised training with labeled data)
        # if self.iter < self.cfg.SEMISUPNET.BURN_UP_STEP:
        #
        #     # input both strong and weak supervised data into model
        #     label_data_q.extend(label_data_k)
        #     record_dict, _, _, _ = self.model(
        #         label_data_q, branch="supervised")
        #
        #     # weight losses
        #     loss_dict = {}
        #     for key in record_dict.keys():
        #         if key[:4] == "loss":
        #             loss_dict[key] = record_dict[key] * 1
        #     losses = sum(loss_dict.values())
        #
        # else:
        #     if self.iter == self.cfg.SEMISUPNET.BURN_UP_STEP:
        #         # update copy the the whole model
        #         self._update_teacher_model(keep_rate=0.00)
        #         # self.model.build_discriminator()
        #
        #     elif (
        #         self.iter - self.cfg.SEMISUPNET.BURN_UP_STEP
        #     ) % self.cfg.SEMISUPNET.TEACHER_UPDATE_ITER == 0:
        #         self._update_teacher_model(
        #             keep_rate=self.cfg.SEMISUPNET.EMA_KEEP_RATE)
        #
        #     record_dict = {}
        #
        #     ######################## For probe #################################
        #     # import pdb; pdb. set_trace()
        #     gt_unlabel_k = self.get_label(unlabel_data_k)
        #     # gt_unlabel_q = self.get_label_test(unlabel_data_q)
        #
        #
        #     #  0. remove unlabeled data labels
        #     unlabel_data_q = self.remove_label(unlabel_data_q)
        #     unlabel_data_k = self.remove_label(unlabel_data_k)
        #
        #     #  1. generate the pseudo-label using teacher model
        #     with torch.no_grad():
        #         (
        #             _,
        #             proposals_rpn_unsup_k,
        #             proposals_roih_unsup_k,
        #             _,
        #         ) = self.model_teacher(unlabel_data_k, branch="unsup_data_weak")
        #
        #     ######################## For probe #################################
        #     # import pdb; pdb. set_trace()
        #
        #     # probe_metrics = ['compute_fp_gtoutlier', 'compute_num_box']
        #     # probe_metrics = ['compute_num_box']
        #     # analysis_pred, _ = self.probe.compute_num_box(gt_unlabel_k,proposals_roih_unsup_k,'pred')
        #     # record_dict.update(analysis_pred)
        #     ######################## For probe END #################################
        #
        #     #  2. Pseudo-labeling
        #     cur_threshold = self.cfg.SEMISUPNET.BBOX_THRESHOLD
        #
        #     joint_proposal_dict = {}
        #     joint_proposal_dict["proposals_rpn"] = proposals_rpn_unsup_k
        #     #Process pseudo labels and thresholding
        #     (
        #         pesudo_proposals_rpn_unsup_k,
        #         nun_pseudo_bbox_rpn,
        #     ) = self.process_pseudo_label(
        #         proposals_rpn_unsup_k, cur_threshold, "rpn", "thresholding"
        #     )
        #     # analysis_pred, _ = self.probe.compute_num_box(gt_unlabel_k,pesudo_proposals_rpn_unsup_k,'pred',True)
        #     # record_dict.update(analysis_pred)
        #
        #     joint_proposal_dict["proposals_pseudo_rpn"] = pesudo_proposals_rpn_unsup_k
        #     # Pseudo_labeling for ROI head (bbox location/objectness)
        #     pesudo_proposals_roih_unsup_k, _ = self.process_pseudo_label(
        #         proposals_roih_unsup_k, cur_threshold, "roih", "thresholding"
        #     )
        #     joint_proposal_dict["proposals_pseudo_roih"] = pesudo_proposals_roih_unsup_k
        #
        #     # 3. add pseudo-label to unlabeled data
        #
        #     unlabel_data_q = self.add_label(
        #         unlabel_data_q, joint_proposal_dict["proposals_pseudo_roih"]
        #     )
        #     unlabel_data_k = self.add_label(
        #         unlabel_data_k, joint_proposal_dict["proposals_pseudo_roih"]
        #     )
        #
        #     all_label_data = label_data_q + label_data_k
        #     all_unlabel_data = unlabel_data_q
        #
        #     # 4. input both strongly and weakly augmented labeled data into student model
        #     record_all_label_data, _, _, _ = self.model(
        #         all_label_data, branch="supervised"
        #     )
        #     record_dict.update(record_all_label_data)
        #
        #     # 5. input strongly augmented unlabeled data into model
        #     record_all_unlabel_data, _, _, _ = self.model(
        #         all_unlabel_data, branch="supervised_target"
        #     )
        #     new_record_all_unlabel_data = {}
        #     for key in record_all_unlabel_data.keys():
        #         new_record_all_unlabel_data[key + "_pseudo"] = record_all_unlabel_data[
        #             key
        #         ]
        #     record_dict.update(new_record_all_unlabel_data)
        #
        #     # 6. input weakly labeled data (source) and weakly unlabeled data (target) to student model
        #     # give sign to the target data
        #
        #     for i_index in range(len(unlabel_data_k)):
        #         # unlabel_data_item = {}
        #         for k, v in unlabel_data_k[i_index].items():
        #             # label_data_k[i_index][k + "_unlabeled"] = v
        #             label_data_k[i_index][k + "_unlabeled"] = v
        #         # unlabel_data_k[i_index] = unlabel_data_item
        #
        #     all_domain_data = label_data_k
        #     # all_domain_data = label_data_k + unlabel_data_k
        #     record_all_domain_data, _, _, _ = self.model(all_domain_data, branch="domain")
        #     record_dict.update(record_all_domain_data)
        #
        #
        #     # weight losses
        #     loss_dict = {}
        #     for key in record_dict.keys():
        #         if key.startswith("loss"):
        #             if key == "loss_rpn_loc_pseudo" or key == "loss_box_reg_pseudo":
        #                 # pseudo bbox regression <- 0
        #                 loss_dict[key] = record_dict[key] * 0
        #             elif key[-6:] == "pseudo":  # unsupervised loss
        #                 loss_dict[key] = (
        #                     record_dict[key] *
        #                     self.cfg.SEMISUPNET.UNSUP_LOSS_WEIGHT
        #                 )
        #             elif (
        #                 key == "loss_D_img_s" or key == "loss_D_img_t"
        #             ):  # set weight for discriminator
        #                 # import pdb
        #                 # pdb.set_trace()
        #                 loss_dict[key] = record_dict[key] * self.cfg.SEMISUPNET.DIS_LOSS_WEIGHT #Need to modify defaults and yaml
        #             else:  # supervised loss
        #                 loss_dict[key] = record_dict[key] * 1
        #
        #     losses = sum(loss_dict.values())
        #
        # metrics_dict = record_dict
        # metrics_dict["data_time"] = data_time
        # self._write_metrics(metrics_dict)
        #
        # self.optimizer.zero_grad()
        # losses.backward()
        # self.optimizer.step()

    def _write_metrics(self, metrics_dict: dict):
        metrics_dict = {
            k: v.detach().cpu().item() if isinstance(v, torch.Tensor) else float(v)
            for k, v in metrics_dict.items()
        }

        # gather metrics among all workers for logging
        # This assumes we do DDP-style training, which is currently the only
        # supported method in detectron2.
        all_metrics_dict = comm.gather(metrics_dict)
        # all_hg_dict = comm.gather(hg_dict)

        if comm.is_main_process():
            if "data_time" in all_metrics_dict[0]:
                # data_time among workers can have high variance. The actual latency
                # caused by data_time is the maximum among workers.
                data_time = np.max([x.pop("data_time")
                                    for x in all_metrics_dict])
                self.storage.put_scalar("data_time", data_time)

            # average the rest metrics
            metrics_dict = {
                k: np.mean([x[k] for x in all_metrics_dict])
                for k in all_metrics_dict[0].keys()
            }

            # append the list
            loss_dict = {}
            for key in metrics_dict.keys():
                if key[:4] == "loss":
                    loss_dict[key] = metrics_dict[key]

            total_losses_reduced = sum(loss for loss in loss_dict.values())

            self.storage.put_scalar("total_loss", total_losses_reduced)
            if len(metrics_dict) > 1:
                self.storage.put_scalars(**metrics_dict)
    @torch.no_grad()
    def _update_student_model(self, keep_rate=0.999):
        if comm.get_world_size() > 1:
            student_model_dict = {
                'module.' + key: value for key, value in self.model_teacher.state_dict().items()
            }
            # for key, value in self.model_teacher.state_dict().items():
            #     print(key)
        else:
            student_model_dict = self.model_teacher.state_dict()

        new_teacher_dict = OrderedDict()
        for key, value in self.model.state_dict().items():
            # print(key,"woshishabi")
            if key in student_model_dict.keys():
                new_teacher_dict[key] = (
                        student_model_dict[key] *
                        (1 - keep_rate) + value * keep_rate
                )
            else:
                raise Exception("{} is not found in student model".format(key))

        self.model.load_state_dict(new_teacher_dict)
    @torch.no_grad()
    def _update_teacher_model(self, keep_rate=0.999):
        if comm.get_world_size() > 1:
            student_model_dict = {
                key[7:]: value for key, value in self.model.state_dict().items()
            }
        else:
            student_model_dict = self.model.state_dict()

        new_teacher_dict = OrderedDict()
        for key, value in self.model_teacher.state_dict().items():
            if key in student_model_dict.keys():
                new_teacher_dict[key] = (
                        student_model_dict[key] *
                        (1 - keep_rate) + value * keep_rate
                )
            else:
                raise Exception("{} is not found in student model".format(key))

        self.model_teacher.load_state_dict(new_teacher_dict)

    @torch.no_grad()
    def _update_teacher_ds_model(self, model_ds,keep_rate=0.9996):
        if comm.get_world_size() > 1:
            student_model_dict = {
                key: value for key, value in self.model_teacher.state_dict().items()
            }
            # for key, value in self.model_teacher.state_dict().items():
            #     print(key)
        else:
            student_model_dict = self.model_teacher.state_dict()

        new_teacher_dict = OrderedDict()
        for key, value in model_ds.state_dict().items():
            # print(key)
            if key in student_model_dict.keys():
                new_teacher_dict[key] = (
                        student_model_dict[key] *
                        (1 - keep_rate) + value * keep_rate
                )
            else:
                raise Exception("{} is not found in student model".format(key))

        model_ds.load_state_dict(new_teacher_dict)

    @torch.no_grad()
    def _update_teacher_dp_model(self,model_dp, keep_rate=0.9996):
        if comm.get_world_size() > 1:
            student_model_dict = {
                key: value for key, value in self.model_teacher.state_dict().items()
            }
        else:
            student_model_dict = self.model_teacher.state_dict()

        new_teacher_dict = OrderedDict()
        for key, value in model_dp.state_dict().items():
            if key in student_model_dict.keys():
                new_teacher_dict[key] = (
                        student_model_dict[key] *
                        (1 - keep_rate) + value * keep_rate
                )
            else:
                raise Exception("{} is not found in student model".format(key))

        model_dp.load_state_dict(new_teacher_dict)

    @torch.no_grad()
    def _copy_main_model(self):
        # initialize all parameters
        if comm.get_world_size() > 1:
            rename_model_dict = {
                key[7:]: value for key, value in self.model.state_dict().items()
            }
            self.model_teacher.load_state_dict(rename_model_dict)
        else:
            self.model_teacher.load_state_dict(self.model.state_dict())

    @classmethod
    def build_test_loader(cls, cfg, dataset_name):
        return build_detection_test_loader(cfg, dataset_name)

    def build_hooks(self):
        cfg = self.cfg.clone()
        cfg.defrost()
        cfg.DATALOADER.NUM_WORKERS = 0  # save some memory and time for PreciseBN

        ret = [
            hooks.IterationTimer(),
            hooks.LRScheduler(self.optimizer, self.scheduler),
            hooks.PreciseBN(
                # Run at the same freq as (but before) evaluation.
                cfg.TEST.EVAL_PERIOD,
                self.model,
                # Build a new data loader to not affect training
                self.build_train_loader(cfg),
                cfg.TEST.PRECISE_BN.NUM_ITER,
            )
            if cfg.TEST.PRECISE_BN.ENABLED and get_bn_modules(self.model)
            else None,
        ]

        # Do PreciseBN before checkpointer, because it updates the model and need to
        # be saved by checkpointer.
        # This is not always the best: if checkpointing has a different frequency,
        # some checkpoints may have more precise statistics than others.
        if comm.is_main_process():
            ret.append(
                hooks.PeriodicCheckpointer(
                    self.checkpointer, cfg.SOLVER.CHECKPOINT_PERIOD
                )
            )

        def test_and_save_results_student():
            self._last_eval_results_student = self.test(self.cfg, self.model)
            _last_eval_results_student = {
                k + "_student": self._last_eval_results_student[k]
                for k in self._last_eval_results_student.keys()
            }
            return _last_eval_results_student

        def test_and_save_results_teacher():
            self._last_eval_results_teacher = self.test(
                self.cfg, self.model_teacher)
            return self._last_eval_results_teacher

        ret.append(hooks.EvalHook(cfg.TEST.EVAL_PERIOD,
                                  test_and_save_results_student))
        ret.append(hooks.EvalHook(cfg.TEST.EVAL_PERIOD,
                                  test_and_save_results_teacher))

        if comm.is_main_process():
            # run writers in the end, so that evaluation metrics are written
            ret.append(hooks.PeriodicWriter(self.build_writers(), period=20))
        return ret

