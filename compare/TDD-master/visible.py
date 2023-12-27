#!/usr/bin/env python3
# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved
import argparse
import glob
import multiprocessing as mp
import os
import time
import cv2
import tqdm
from detectron2.engine import default_argument_parser, default_setup, launch
from detectron2.config import get_cfg
from detectron2.config import get_cfg
from detectron2.data.detection_utils import read_image
from detectron2.utils.logger import setup_logger
from adapteacher import add_ateacher_config
from predictor import VisualizationDemo
from adapteacher.engine.trainer_baseline import ATeacherTrainer, BaselineTrainer
# constants
from adapteacher.modeling.meta_arch.ts_ensemble import EnsembleTSModel
from detectron2.checkpoint import DetectionCheckpointer
WINDOW_NAME = "COCO detections"
import detectron2.utils.comm as comm
from detectron2.checkpoint import DetectionCheckpointer
from detectron2.config import get_cfg
from detectron2.engine import default_argument_parser, default_setup, launch
from detectron2.engine import HookBase
from adapteacher import add_ateacher_config
from adapteacher.engine.trainer_baseline import ATeacherTrainer, BaselineTrainer
import os
# hacky way to register
from adapteacher.modeling.meta_arch.rcnn_baseline import TwoStagePseudoLabGeneralizedRCNN, DAobjTwoStagePseudoLabGeneralizedRCNN
from adapteacher.modeling.meta_arch.vgg import build_vgg_backbone  # noqa
from adapteacher.modeling.proposal_generator.rpn import PseudoLabRPN
from adapteacher.modeling.roi_heads.roi_heads import StandardROIHeadsPseudoLab
import adapteacher.data.datasets.builtin
import numpy as np
import random
from adapteacher.modeling.meta_arch.ts_ensemble import EnsembleTSModel
import torch.multiprocessing

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


def main(args):
    cfg = setup(args)
    if cfg.SEMISUPNET.Trainer == "ateacher":
        Trainer = ATeacherTrainer
    elif cfg.SEMISUPNET.Trainer == "baseline":
        Trainer = BaselineTrainer
    else:
        raise ValueError("Trainer Name is not found.")

    #可视化结果部分
    if True:
        if cfg.SEMISUPNET.Trainer == "ateacher":
            model = Trainer.build_model(cfg)
            model_teacher = Trainer.build_model(cfg)
            # ensem_ts_model = EnsembleTSModel(model_teacher, model)
            ensem_ts_model = EnsembleTSModel(model, model_teacher)

            DetectionCheckpointer(
                ensem_ts_model, save_dir=cfg.OUTPUT_DIR
            ).resume_or_load(cfg.MODEL.WEIGHTS, resume=args.resume)
            #可视化器，这里跳转到predictor
            demo = VisualizationDemo(cfg)
        if True:
            # if len(args.input) == 1:
            #     args.input = glob.glob(os.path.expanduser(args.input[0]))
            #     assert args.input, "The input path(s) was not found"
            args.input = ['datasets/sonar_semi/semi/*']
            # args.input = ['datasets/547/pic/*']
            args.input = glob.glob(os.path.expanduser(args.input[0]))
            for path in tqdm.tqdm(args.input):
                # use PIL, to be consistent with evaluation
                img = read_image(path, format="BGR")
                start_time = time.time()
                logger = setup_logger()
                logger.info("Arguments: " + str(args))
                predictions, visualized_output = demo.run_on_image(img)
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
                args.output = '/home/jhvision-2/cvdisk/wcw/raw/adaptive_teacher/visible_/'
                if args.output:
                    if os.path.isdir(args.output):
                        # if True:
                        assert os.path.isdir(args.output), args.output
                        out_filename = os.path.join(args.output, os.path.basename(path))
                    else:
                        assert len(args.input) == 1, "Please specify a directory with args.output"
                        out_filename = args.output
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
            # ensem_ts_model = EnsembleTSModel(model_teacher, model)
            ensem_ts_model = EnsembleTSModel(model, model_teacher)

            DetectionCheckpointer(
                ensem_ts_model, save_dir=cfg.OUTPUT_DIR
            ).resume_or_load(cfg.MODEL.WEIGHTS, resume=args.resume)
            # res = Trainer.test(cfg, ensem_ts_model.modelTeacher)
            res = Trainer.test(cfg, ensem_ts_model.modelStudent)

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





# 设置随机数种子
# setup_seed(1234)


if __name__ == "__main__":



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
        dist_url=args.dist_url,
        args=(args,),
    )
