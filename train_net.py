#!/usr/bin/env python3
# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved
import os
os.environ['CUDA_VISIBLE_DEVICES']='1,2'


import random
import numpy as np
import torch
import detectron2.utils.comm as comm
from detectron2.checkpoint import DetectionCheckpointer
from detectron2.config import get_cfg
from detectron2.engine import default_argument_parser, default_setup, launch
from detectron2.engine import HookBase
from adapteacher import add_ateacher_config
from adapteacher.engine.trainer import ATeacherTrainer, BaselineTrainer

# hacky way to register
from adapteacher.modeling.meta_arch.rcnn import TwoStagePseudoLabGeneralizedRCNN, DAobjTwoStagePseudoLabGeneralizedRCNN
from adapteacher.modeling.meta_arch.vgg import build_vgg_backbone  # noqa
from adapteacher.modeling.proposal_generator.rpn import PseudoLabRPN
from adapteacher.modeling.roi_heads.roi_heads import StandardROIHeadsPseudoLab
import adapteacher.data.datasets.builtin
from predictor import VisualizationDemo
from adapteacher.modeling.meta_arch.ts_ensemble import EnsembleTSModel
import torch.multiprocessing
from detectron2.data.detection_utils import read_image
from detectron2.utils.logger import setup_logger
import random
import numpy as np
import torch
import glob
import time
import tqdm
import cv2
WINDOW_NAME = "COCO detections"
import multiprocessing as mp

# os.environ["CUDA_VISIBLE_DEVICES"]="3"

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
class BestCheckpointer(HookBase):
  def __init__(self):
      super().__init__()

  def after_step(self):
    # No way to use **kwargs

    ##ONly do this analys when trainer.iter is divisle by checkpoint_epochs
    curr_val = self.trainer.storage.latest().get('bbox/AP50', 0)
    '''这里做了小改动'''
    import math
    if type(curr_val) != int:
        curr_val = curr_val[0]
        if math.isnan(curr_val):
            curr_val = 0

    try:
        _ = self.trainer.storage.history('max_bbox/AP50')
    except:
        self.trainer.storage.put_scalar('max_bbox/AP50', curr_val)

    max_val = self.trainer.storage.history('max_bbox/AP50')._data[-1][0]

    #print(curr_val, max_val)
    if curr_val > max_val:
        print("\n%s > %s要存！！\n"%(curr_val,max_val))
        self.trainer.storage.put_scalar('max_bbox/AP50', curr_val)
        self.trainer.checkpointer.save("model_best")
        #self.step(self.trainer.iter)


def main(args):
    cfg = setup(args)
    if cfg.SEMISUPNET.Trainer == "ateacher":
        Trainer = ATeacherTrainer
    elif cfg.SEMISUPNET.Trainer == "baseline":
        Trainer = BaselineTrainer
    else:
        raise ValueError("Trainer Name is not found.")
    # 可视化
    if True:
        if cfg.SEMISUPNET.Trainer == "ateacher":
            model = Trainer.build_model(cfg)
            model_teacher = Trainer.build_model(cfg)
            # ensem_ts_model = EnsembleTSModel(model_teacher, model)
            ensem_ts_model = EnsembleTSModel(model, model_teacher)
    
            DetectionCheckpointer(
                ensem_ts_model, save_dir=cfg.OUTPUT_DIR
            ).resume_or_load(cfg.MODEL.WEIGHTS, resume=args.resume)
            demo = VisualizationDemo(cfg)
        if True:
            # if len(args.input) == 1:
            #     args.input = glob.glob(os.path.expanduser(args.input[0]))
            #     assert args.input, "The input path(s) was not found"
            # args.input = ['datasets/changjiangkou/val/*']
            args.input = ['/home/shu3090/wcw/datasets/sonar_semi/semi/*']
            args.input = glob.glob(os.path.expanduser(args.input[0]))
            for path in tqdm.tqdm(args.input):
                # use PIL, to be consistent with evaluation
                img = read_image(path, format="BGR")
                start_time = time.time()
                logger = setup_logger()
                logger.info("Arguments: " + str(args))
                predictions, visualized_output = demo.run_on_image(img)
                # predictions = predictions[0]
                # val_map = predictions['instances'].scores >= 0.5
                # predictions = predictions['instances'][val_map]
                logger.info(
                    "{}: {} in {:.2f}s".format(
                        path,
                        "detected {} instances".format(len(predictions["instances"]))
                        # "detected {} instances".format(len(predictions))
                        if "instances" in predictions
                        # if len(predictions)>0
                        else "finished",
                        time.time() - start_time,
                    )
                )
                args.output = './SSS_VISIBLE_80'
                
                if args.output:
                    if os.path.isdir(args.output):
                        # if True:
                        assert os.path.isdir(args.output), args.output
                        out_filename = os.path.join(args.output, os.path.basename(path))
                    else:
                        assert len(args.input) == 1, "Please specify a directory with args.output"
                        out_filename = args.output
                    if visualized_output is None:
                        continue
                    print(type(img))
                    visualized_output.save(out_filename)
                else:
                    cv2.namedWindow(WINDOW_NAME, cv2.WINDOW_NORMAL)
                    cv2.imshow(WINDOW_NAME, visualized_output.get_image()[:, :, ::-1])
                    if cv2.waitKey(0) == 27:
                        break  # esc to quit
    if args.eval_only:
        if cfg.SEMISUPNET.Trainer == "ateacher":
            model = Trainer.build_model(cfg)
            model_teacher = Trainer.build_model(cfg)
            ensem_ts_model = EnsembleTSModel(model_teacher, model)

            DetectionCheckpointer(
                ensem_ts_model, save_dir=cfg.OUTPUT_DIR
            ).resume_or_load(cfg.MODEL.WEIGHTS, resume=args.resume)
            res = Trainer.test(cfg, ensem_ts_model.modelTeacher)

        else:
            model = Trainer.build_model(cfg)
            DetectionCheckpointer(model, save_dir=cfg.OUTPUT_DIR).resume_or_load(
                cfg.MODEL.WEIGHTS, resume=args.resume
            )
            res = Trainer.test(cfg, model)
        return res

    os.makedirs(cfg.OUTPUT_DIR, exist_ok=True)
    trainer = Trainer(cfg)
    trainer.register_hooks([BestCheckpointer()])
    trainer.resume_or_load(resume=args.resume)

    return trainer.train()

def setup_seed(seed):
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)
    torch.backends.cudnn.deterministic = True


if __name__ == "__main__":
    setup_seed(608123)
    args = default_argument_parser().parse_args()

    # export:
    # PYTHONWARNINGS = 'ignore:semaphore_tracker:UserWarning'
    torch.multiprocessing.set_sharing_strategy('file_system')
    print("Command Line Args:", args)
    launch(
        main,
        args.num_gpus,
        num_machines=args.num_machines,
        machine_rank=args.machine_rank,
        # dist_url=args.dist_url,
        dist_url="tcp://127.0.0.1:50153",
        args=(args,),
    )
#
# #!/usr/bin/env python3
# # Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved
# import os
# import glob
# import glob
# import multiprocessing as mp
# from predictor import VisualizationDemo
# import time
# from detectron2.data.detection_utils import read_image
# from detectron2.utils.logger import setup_logger
# WINDOW_NAME = "COCO detections"
# import cv2
# import tqdm
# import random
# import collections
# import numpy as np
# import datetime
# import torch
# import logging
# from contextlib import contextmanager
# import detectron2.utils.comm as comm
# from detectron2.checkpoint import DetectionCheckpointer
# from detectron2.config import get_cfg
# from detectron2.engine import default_argument_parser, default_setup, launch
# from detectron2.engine import HookBase
# from adapteacher import add_ateacher_config
# from adapteacher.engine.trainer import ATeacherTrainer, BaselineTrainer
# from detectron2.utils.comm import get_world_size, is_main_process
# # hacky way to register
# from adapteacher.modeling.meta_arch.rcnn import TwoStagePseudoLabGeneralizedRCNN, DAobjTwoStagePseudoLabGeneralizedRCNN
# from adapteacher.modeling.meta_arch.vgg import build_vgg_backbone  # noqa
# from adapteacher.modeling.proposal_generator.rpn import PseudoLabRPN
# from adapteacher.modeling.roi_heads.roi_heads import StandardROIHeadsPseudoLab
# import adapteacher.data.datasets.builtin
# from detectron2.structures import Boxes, ImageList, Instances, pairwise_iou
# from detectron2.evaluation import (
#     DatasetEvaluator,
#     print_csv_format,
#     verify_results,
# )
# from collections import OrderedDict
#
# from detectron2.utils.logger import log_every_n_seconds
# import time
# from adapteacher.modeling.meta_arch.ts_ensemble import EnsembleTSModel
# import torch.multiprocessing
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
#             # b = [int(label), (float(score) - thr) * weights[t], float(box_part[0]), float(box_part[1]), float(box_part[2]),
#             #      float(box_part[3])]
#             b = [int(label), (float(score)-thr) * weights[t], float(box_part[0]), float(box_part[1]),
#                  float(box_part[2]),
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
#
# def get_weighted_box(boxes,thr, conf_type='avg'):
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
#     # box_area1 = int((boxes[0][5] - boxes[0][3]) * (boxes[0][4] - boxes[0][2]))
#     # box_area2 = int((boxes[1][5] - boxes[1][3]) * (boxes[1][4] - boxes[1][2]))
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
#         box[1] = conf / len(boxes) + thr
#     elif conf_type == 'max':
#         box[1] = np.array(conf_list).max()
#     box[2:] /= (conf )
#     return box
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
#         #filter boxes which score > thr
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
#                 weighted_boxes[index] = get_weighted_box(new_boxes[index], skip_box_thr,conf_type)
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
#
# class DatasetEvaluators(DatasetEvaluator):
#     """
#     Wrapper class to combine multiple :class:`DatasetEvaluator` instances.
#
#     This class dispatches every evaluation call to
#     all of its :class:`DatasetEvaluator`.
#     """
#
#     def __init__(self, evaluators):
#         """
#         Args:
#             evaluators (list): the evaluators to combine.
#         """
#         super().__init__()
#         self._evaluators = evaluators
#
#     def reset(self):
#         for evaluator in self._evaluators:
#             evaluator.reset()
#
#     def process(self, inputs, outputs):
#         for evaluator in self._evaluators:
#             evaluator.process(inputs, outputs)
#
#     def evaluate(self):
#         results = collections.OrderedDict()
#         for evaluator in self._evaluators:
#             result = evaluator.evaluate()
#             if is_main_process() and result is not None:
#                 for k, v in result.items():
#                     assert (
#                         k not in results
#                     ), "Different evaluators produce results with the same key {}".format(k)
#                     results[k] = v
#         return results
# def WBF(output_t,output_s,iou_thr = 0.5,skip_box_thr = 0.05,weights = [1,1]):
#
#     image_size = output_t[0]['instances'].image_size
#
#     boxes_list,scores_list,labels_list = [],[],[]
#
#     box_list_t = output_t[0]['instances'].pred_boxes.tensor
#     scores_list_t = output_t[0]['instances'].scores
#     classes_list_t = output_t[0]['instances'].pred_classes
#
#     box_list_s = output_s[0]['instances'].pred_boxes.tensor
#     scores_list_s = output_s[0]['instances'].scores
#     classes_list_s = output_s[0]['instances'].pred_classes
#
#     boxes_list.append(box_list_t)
#     boxes_list.append(box_list_s)
#     scores_list.append(scores_list_t)
#     scores_list.append(scores_list_s)
#     labels_list.append(classes_list_t)
#     labels_list.append(classes_list_s)
#     boxes, scores, labels = weighted_boxes_fusion(boxes_list, scores_list, labels_list, weights=weights,
#                                                   iou_thr=iou_thr, skip_box_thr=skip_box_thr)
#     result = Instances(image_size)
#     valid = scores>=0
#     boxes = Boxes(torch.tensor(boxes[valid]))
#     boxes.clip(image_size)
#     result.pred_boxes = boxes
#     result.scores = torch.tensor(scores[valid])
#     # print(scores)
#     result.pred_classes = torch.tensor(labels[valid])
#     return [{'instances':result}]
#
# def process_pseudo_label(proposals_rpn_unsup_k, cur_threshold, proposal_type, psedo_label_method=""):
#         list_instances = []
#         num_proposal_output = 0.0
#         for proposal_bbox_inst in proposals_rpn_unsup_k:
#             # thresholding
#             if psedo_label_method == "thresholding":
#                 proposal_bbox_inst = threshold_bbox(
#                     proposal_bbox_inst, thres=cur_threshold, proposal_type=proposal_type
#                 )
#             # elif psedo_label_method == "NMS":
#             #     proposal_bbox_inst = NonMaxSuppression(
#             #         proposal_bbox_inst, confi_thres=cur_threshold, proposal_type=proposal_type
#             #     )
#
#             else:
#                 raise ValueError("Unkown pseudo label boxes methods")
#             num_proposal_output += len(proposal_bbox_inst)
#             list_instances.append(proposal_bbox_inst)
#         num_proposal_output = num_proposal_output / len(proposals_rpn_unsup_k)
#         return list_instances, num_proposal_output
#
# def threshold_bbox(proposal_bbox_inst, thres=0.7, proposal_type="roih"):
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
#             valid_map = proposal_bbox_inst['instances'].scores > thres
#
#             # create instances containing boxes and gt_classes
#             image_shape = proposal_bbox_inst['instances'].image_size
#             new_proposal_inst = Instances(image_shape)
#
#             # create box
#             new_bbox_loc = proposal_bbox_inst['instances'].pred_boxes.tensor[valid_map, :]
#             new_boxes = Boxes(new_bbox_loc)
#
#             # add boxes to instances
#             new_proposal_inst.gt_boxes = new_boxes
#             new_proposal_inst.gt_classes = proposal_bbox_inst['instances'].pred_classes[valid_map]
#             new_proposal_inst.scores = proposal_bbox_inst['instances'].scores[valid_map]
#
#         return new_proposal_inst
#
# @contextmanager
# def inference_context(model):
#     """
#     A context where the model is temporarily changed to eval mode,
#     and restored to previous mode afterwards.
#
#     Args:
#         model: a torch Module
#     """
#     training_mode = model.training
#     model.eval()
#     yield
#     model.train(training_mode)
#
# @torch.no_grad()
# def _fusion_teacher_model(model_1, model_2, keep_rate=0.5):
#     if comm.get_world_size() > 1:
#         model_1_dict = {
#             key[7:]: value for key, value in model_1.state_dict().items()
#         }
#     else:
#         model_1_dict = model_1.state_dict()
#         # model_2_dict = model_2.state_dict()
#
#     model_2_dict = OrderedDict()
#     for key, value in model_2.state_dict().items():
#         if key in model_1_dict.keys():
#             model_2_dict[key] = (
#                     model_1_dict[key] *
#                     (1 - keep_rate) + value * keep_rate
#             )
#         else:
#             raise Exception("{} is not found in student model".format(key))
#
#     model_2.load_state_dict(model_2_dict)
#     return model_2
#
# def inference_on_dataset(model_s,model_p, data_loader, evaluator):
#     """
#     Run model on the data_loader and evaluate the metrics with evaluator.
#     Also benchmark the inference speed of `model.forward` accurately.
#     The model will be used in eval mode.
#
#     Args:
#         model (nn.Module): a module which accepts an object from
#             `data_loader` and returns some outputs. It will be temporarily set to `eval` mode.
#
#             If you wish to evaluate a model in `training` mode instead, you can
#             wrap the given model and override its behavior of `.eval()` and `.train()`.
#         data_loader: an iterable object with a length.
#             The elements it generates will be the inputs to the model.
#         evaluator (DatasetEvaluator): the evaluator to run. Use `None` if you only want
#             to benchmark, but don't want to do any evaluation.
#
#     Returns:
#         The return value of `evaluator.evaluate()`
#     """
#     num_devices = get_world_size()
#     logger = logging.getLogger(__name__)
#     logger.info("Start inference on {} images".format(len(data_loader)))
#
#     total = len(data_loader)  # inference data loader must have a fixed length
#     if evaluator is None:
#         # create a no-op evaluator
#         evaluator = DatasetEvaluators([])
#     evaluator.reset()
#
#     num_warmup = min(5, total - 1)
#     start_time = time.perf_counter()
#     total_compute_time = 0
#     EMA_rate = 0.5
#
#     with inference_context(model_p),inference_context(model_s), torch.no_grad():
#
#         model_123 = _fusion_teacher_model(model_p.modelTeacher, model_s.modelTeacher, EMA_rate)
#
#         for idx, inputs in enumerate(data_loader):
#             if idx == num_warmup:
#                 start_time = time.perf_counter()
#                 total_compute_time = 0
#
#             start_compute_time = time.perf_counter()
#
#             # todo: conbination output
#
#             # outputs = model_s.modelTeacher(inputs)
#             # #
#             # output_p = model_p.modelTeacher(inputs)
#             # outputs = WBF(output_s, output_p,iou_thr=0.6,skip_box_thr=0.05,weights = [1,1])
#
#
#             #oral
#             outputs = model_123(inputs)
#
#             if torch.cuda.is_available():
#                 torch.cuda.synchronize()
#             total_compute_time += time.perf_counter() - start_compute_time
#             evaluator.process(inputs, outputs)
#
#             iters_after_start = idx + 1 - num_warmup * int(idx >= num_warmup)
#             seconds_per_img = total_compute_time / iters_after_start
#             if idx >= num_warmup * 2 or seconds_per_img > 5:
#                 total_seconds_per_img = (time.perf_counter() - start_time) / iters_after_start
#                 eta = datetime.timedelta(seconds=int(total_seconds_per_img * (total - idx - 1)))
#                 log_every_n_seconds(
#                     logging.INFO,
#                     "Inference done {}/{}. {:.4f} s / img. ETA={}".format(
#                         idx + 1, total, seconds_per_img, str(eta)
#                     ),
#                     n=5,
#                 )
#
#     # Measure the time only for this worker (before the synchronization barrier)
#     total_time = time.perf_counter() - start_time
#     total_time_str = str(datetime.timedelta(seconds=total_time))
#     # NOTE this format is parsed by grep
#     logger.info(
#         "Total inference time: {} ({:.6f} s / img per device, on {} devices)".format(
#             total_time_str, total_time / (total - num_warmup), num_devices
#         )
#     )
#     total_compute_time_str = str(datetime.timedelta(seconds=int(total_compute_time)))
#     logger.info(
#         "Total inference pure compute time: {} ({:.6f} s / img per device, on {} devices)".format(
#             total_compute_time_str, total_compute_time / (total - num_warmup), num_devices
#         )
#     )
#
#     results = evaluator.evaluate()
#     # An evaluator may return None when not in main process.
#     # Replace it by an empty dict instead to make it easier for downstream code to handle
#     if results is None:
#         results = {}
#     return results
#
#
# # @classmethod
# def test(cls, cfg, model_s,model_p, evaluators=None):
#     """
#     Args:
#         cfg (CfgNode):
#         model (nn.Module):
#         evaluators (list[DatasetEvaluator] or None): if None, will call
#             :meth:`build_evaluator`. Otherwise, must have the same length as
#             ``cfg.DATASETS.TEST``.
#
#     Returns:
#         dict: a dict of result metrics
#     """
#     logger = logging.getLogger(__name__)
#     if isinstance(evaluators, DatasetEvaluator):
#         evaluators = [evaluators]
#     if evaluators is not None:
#         assert len(cfg.DATASETS.TEST) == len(evaluators), "{} != {}".format(
#             len(cfg.DATASETS.TEST), len(evaluators)
#         )
#
#     results = collections.OrderedDict()
#     for idx, dataset_name in enumerate(cfg.DATASETS.TEST):
#         data_loader = cls.build_test_loader(cfg, dataset_name)
#         # When evaluators are passed in as arguments,
#         # implicitly assume that evaluators can be created before data_loader.
#         if evaluators is not None:
#             evaluator = evaluators[idx]
#         else:
#             try:
#                 evaluator = cls.build_evaluator(cfg, dataset_name)
#             except NotImplementedError:
#                 logger.warn(
#                     "No evaluator found. Use `DefaultTrainer.test(evaluators=)`, "
#                     "or implement its `build_evaluator` method."
#                 )
#                 results[dataset_name] = {}
#                 continue
#         results_i = inference_on_dataset(model_s,model_p, data_loader, evaluator)
#         results[dataset_name] = results_i
#         if comm.is_main_process():
#             assert isinstance(
#                 results_i, dict
#             ), "Evaluator must return a dict on the main process. Got {} instead.".format(
#                 results_i
#             )
#             logger.info("Evaluation results for {} in csv format:".format(dataset_name))
#             print_csv_format(results_i)
#
#     if len(results) == 1:
#         results = list(results.values())[0]
#     return results
#
#
# def setup(args):
#     """
#     Create configs and perform basic setups.
#     """
#     cfg = get_cfg()
#     add_ateacher_config(cfg)
#     cfg.merge_from_file(args.config_file)
#     cfg.merge_from_list(args.opts)
#     cfg.freeze()
#     default_setup(cfg, args)
#     return cfg
#
# def main(args):
#     cfg = setup(args)
#     if cfg.SEMISUPNET.Trainer == "ateacher":
#         Trainer = ATeacherTrainer
#     elif cfg.SEMISUPNET.Trainer == "baseline":
#         Trainer = BaselineTrainer
#     else:
#         raise ValueError("Trainer Name is not found.")
#     # if True:
#     #     if cfg.SEMISUPNET.Trainer == "ateacher":
#     #         model = Trainer.build_model(cfg)
#     #         model_teacher = Trainer.build_model(cfg)
#     #         # ensem_ts_model = EnsembleTSModel(model_teacher, model)
#     #         ensem_ts_model = EnsembleTSModel(model, model_teacher)
#     #
#     #         DetectionCheckpointer(
#     #             ensem_ts_model, save_dir=cfg.OUTPUT_DIR
#     #         ).resume_or_load(cfg.MODEL.WEIGHTS, resume=args.resume)
#     #         demo = VisualizationDemo(cfg)
#     #     if True:
#     #         # if len(args.input) == 1:
#     #         #     args.input = glob.glob(os.path.expanduser(args.input[0]))
#     #         #     assert args.input, "The input path(s) was not found"
#     #         args.input = ['datasets/sonar_semi/semi/*']
#     #         # args.input = ['datasets/547/pic/*']
#     #         args.input = glob.glob(os.path.expanduser(args.input[0]))
#     #         for path in tqdm.tqdm(args.input):
#     #             # use PIL, to be consistent with evaluation
#     #
#     #             img = read_image(path, format="BGR")
#     #             start_time = time.time()
#     #             logger = setup_logger()
#     #             logger.info("Arguments: " + str(args))
#     #             predictions, visualized_output = demo.run_on_image(img)
#     #             # val_map = predictions['instances'].scores >= 0.8
#     #             # predictions = predictions['instances'][val_map]
#     #             logger.info(
#     #                 "{}: {} in {:.2f}s".format(
#     #                     path,
#     #                     "detected {} instances".format(len(predictions["instances"]))
#     #                     # "detected {} instances".format(len(predictions))
#     #                     if "instances" in predictions
#     #                     # if len(predictions)>0
#     #                     else "finished",
#     #                     time.time() - start_time,
#     #                 )
#     #             )
#     #             args.output = "./try/"
#     #             if args.output:
#     #                 if os.path.isdir(args.output):
#     #                     # if True:
#     #                     assert os.path.isdir(args.output), args.output
#     #                     out_filename = os.path.join(args.output, os.path.basename(path))
#     #                 else:
#     #                     assert len(args.input) == 1, "Please specify a directory with args.output"
#     #                     out_filename = args.output
#     #                 if visualized_output!=None:
#     #                     visualized_output.save(out_filename)
#     #             else:
#     #                 cv2.namedWindow(WINDOW_NAME, cv2.WINDOW_NORMAL)
#     #                 cv2.imshow(WINDOW_NAME, visualized_output.get_image()[:, :, ::-1])
#     #                 if cv2.waitKey(0) == 27:
#     #                     break  # esc to quit
#
#     if args.eval_only:
#         if cfg.SEMISUPNET.Trainer == "ateacher":
#             model_ss = Trainer.build_model(cfg)
#             model_teacher_ds = Trainer.build_model(cfg)
#             ensem_ts_model_s = EnsembleTSModel(model_teacher_ds, model_ss)
#             model_sp = Trainer.build_model(cfg)
#             model_teacher_dp = Trainer.build_model(cfg)
#             ensem_ts_model_p = EnsembleTSModel(model_teacher_dp, model_sp)
#
#             DetectionCheckpointer(
#                 ensem_ts_model_s, save_dir=cfg.OUTPUT_DIR
#             ).resume_or_load(cfg.MODEL.WEIGHTS_DS, resume=args.resume)
#             DetectionCheckpointer(
#                 ensem_ts_model_p, save_dir=cfg.OUTPUT_DIR
#             ).resume_or_load(cfg.MODEL.WEIGHTS_DP, resume=args.resume)
#             # res = Trainer.test(cfg, ensem_ts_model.modelTeacher)
#             res = test(Trainer,cfg, ensem_ts_model_s,ensem_ts_model_p)
#
#         else:
#             model = Trainer.build_model(cfg)
#             DetectionCheckpointer(model, save_dir=cfg.OUTPUT_DIR).resume_or_load(
#                 cfg.MODEL.WEIGHTS, resume=args.resume
#             )
#             res = test(cfg, model)
#         return res
#
#     os.makedirs(cfg.OUTPUT_DIR, exist_ok=True)
#     trainer = Trainer(cfg)
#
#     trainer.resume_or_load(resume=args.resume)
#
#     return None
#
# def setup_seed(seed):
#     torch.manual_seed(seed)
#     torch.cuda.manual_seed_all(seed)
#     np.random.seed(seed)
#     random.seed(seed)
#     torch.backends.cudnn.deterministic = True
#
#
# if __name__ == "__main__":
#     setup_seed(608123)
#     args = default_argument_parser().parse_args()
#
#     # export:
#     # PYTHONWARNINGS = 'ignore:semaphore_tracker:UserWarning'
#     torch.multiprocessing.set_sharing_strategy('file_system')
#     print("Command Line Args:", args)
#     launch(
#         main,
#         args.num_gpus,
#         num_machines=args.num_machines,
#         machine_rank=args.machine_rank,
#         dist_url=args.dist_url,
#         args=(args,),
#     )
#
