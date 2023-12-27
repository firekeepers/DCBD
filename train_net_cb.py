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
from adapteacher.engine.trainer_cb import ATeacherTrainer, BaselineTrainer

# hacky way to register
from adapteacher.modeling.meta_arch.rcnn import TwoStagePseudoLabGeneralizedRCNN, DAobjTwoStagePseudoLabGeneralizedRCNN
from adapteacher.modeling.meta_arch.vgg import build_vgg_backbone  # noqa
from adapteacher.modeling.proposal_generator.rpn import PseudoLabRPN
from adapteacher.modeling.roi_heads.roi_heads import StandardROIHeadsPseudoLab
import adapteacher.data.datasets.builtin
import torch.nn as nn
from torch.nn.parallel import DataParallel, DistributedDataParallel
# from adapteacher.modeling.meta_arch.ts_ensemble import EnsembleTSModel
import torch.multiprocessing


class EnsembleTSModel(nn.Module):
    def __init__(self, modelTeacher, modelStudent):
        super(EnsembleTSModel, self).__init__()

        if isinstance(modelTeacher, (DistributedDataParallel, DataParallel)):
            modelTeacher = modelTeacher.module
        # if isinstance(modelTeacher, (DistributedDataParallel, DataParallel)):
        #     modelTeacher_dp = modelTeacher.module
        if isinstance(modelStudent, (DistributedDataParallel, DataParallel)):
            modelStudent = modelStudent.module
        # if isinstance(modelDomain, (DistributedDataParallel, DataParallel)):
        #     modelStudent = modelDomain.module

        self.modelTeacher = modelTeacher
        # self.modelTeacher_dp = modelTeacher_dp
        self.modelStudent = modelStudent

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

    if args.eval_only:
        if cfg.SEMISUPNET.Trainer == "ateacher":
            model = Trainer.build_model(cfg)
            model_teacher = Trainer.build_model(cfg)
            ensem_ts_model = EnsembleTSModel(model_teacher, model)

            DetectionCheckpointer(
                ensem_ts_model, save_dir=cfg.OUTPUT_DIR
            ).resume_or_load(cfg.MODEL.WEIGHTS, resume=args.resume)
            #
            # DetectionCheckpointer(
            #     ensem_ts_model, save_dir=cfg.OUTPUT_DIR
            # ).resume_or_load(cfg.MODEL.WEIGHTS, resume=args.resume)

            res = Trainer.test(cfg, ensem_ts_model.modelTeacher)

        else:
            model = Trainer.build_model(cfg)
            DetectionCheckpointer(model, save_dir=cfg.OUTPUT_DIR).resume_or_load(
                cfg.MODEL.WEIGHTS, resume=args.resume
            )
            res = Trainer.test(cfg, model)
        return res
    # if args.cb_training:

    os.makedirs(cfg.OUTPUT_DIR, exist_ok=True)
    trainer = Trainer(cfg)
    trainer.register_hooks([BestCheckpointer()])
    model_s = Trainer.build_model(cfg)
    model_teacher_ds = Trainer.build_model(cfg)
    model_teacher_dp = Trainer.build_model(cfg)
    ensem_ts_model_ds = EnsembleTSModel(model_teacher_ds, model_s)
    ensem_ts_model_dp = EnsembleTSModel(model_teacher_dp, model_s)
    # DetectionCheckpointer(model_s, save_dir=cfg.OUTPUT_DIR).resume_or_load(
    #     cfg.MODEL.WEIGHTS_s, resume=args.resume
    # )
    DetectionCheckpointer(ensem_ts_model_ds, save_dir=cfg.DS_OUTPUT_DIR).resume_or_load(
        cfg.MODEL.WEIGHTS_DS, resume=args.resume
    )
    DetectionCheckpointer(ensem_ts_model_dp, save_dir=cfg.DP_OUTPUT_DIR).resume_or_load(
        cfg.MODEL.WEIGHTS_DP, resume=args.resume
    )
    trainer.resume_or_load(resume=args.resume)

    return trainer.train(ensem_ts_model_ds.modelTeacher,ensem_ts_model_dp.modelTeacher)

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
        dist_url=args.dist_url,
        args=(args,),
    )

