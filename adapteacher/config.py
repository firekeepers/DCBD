# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved
from detectron2.config import CfgNode as CN


def add_ateacher_config(cfg):
    """
    Add config for semisupnet.
    """
    _C = cfg
    _C.TEST.VAL_LOSS = True

    _C.MODEL.RPN.UNSUP_LOSS_WEIGHT = 1.0
    _C.MODEL.RPN.LOSS = "CrossEntropy"
    _C.MODEL.ROI_HEADS.LOSS = "CrossEntropy"
    _C.MODEL.WEIGHTS_DP = "/home/shu3090/wcw/clipart_IDCC/clipart_0.2grl_0.9996_3unsup_1sup_10000_ini_0.02pseudo_compare/model_best_IDCC_51.13.pth"
    _C.MODEL.WEIGHTS_DS = "/home/shu3090/wcw/clipart_/dp_1/model_best.pth"
    _C.SOLVER.MAX_ROUND = 5

    _C.SOLVER.IMG_PER_BATCH_LABEL_TRAIN = 4
    _C.SOLVER.IMG_PER_BATCH_LABEL_COMPARE = 4
    _C.SOLVER.IMG_PER_BATCH_UNLABEL = 4
    _C.SOLVER.FACTOR_LIST = (1,)
    _C.SOLVER.CHECKPOINT_PERIOD = 5000

    _C.DATASETS.TRAIN_LABEL = ("coco_2017_train",)
    _C.DATASETS.COMPARE_LABEL = ("coco_2017_train",)
    _C.DATASETS.TRAIN_UNLABEL = ("coco_2017_train",)
    _C.DATASETS.CROSS_DATASET = True
    _C.TEST.EVALUATOR = "COCOeval"

    _C.SEMISUPNET = CN()

    # Output dimension of the MLP projector after `res5` block
    _C.SEMISUPNET.MLP_DIM = 128

    # Semi-supervised training
    _C.SEMISUPNET.Trainer = "ateacher"
    _C.SEMISUPNET.BBOX_THRESHOLD = 0.7
    _C.SEMISUPNET.COMPARE_THRESHOLD = 0.7
    _C.SEMISUPNET.PSEUDO_BBOX_SAMPLE = "thresholding"
    _C.SEMISUPNET.TEACHER_UPDATE_ITER = 1
    _C.SEMISUPNET.BURN_UP_STEP = 12000
    _C.SEMISUPNET.EMA_KEEP_RATE = 0.0
    _C.SEMISUPNET.UNSUP_LOSS_WEIGHT = 4.0
    _C.SEMISUPNET.SUP_LOSS_WEIGHT = 0.5
    _C.SEMISUPNET.LOSS_WEIGHT_TYPE = "standard"
    _C.SEMISUPNET.DIS_TYPE = "res4"
    _C.SEMISUPNET.DIS_LOSS_WEIGHT = 0.1
    _C.SEMISUPNET.FUSION_IOU_THR = 0.6
    _C.SEMISUPNET.FUSION_BBOX_THRESHOLD = 0.8
    _C.SEMISUPNET.FUSION_WEIGHT = [1,1]

    _C.SEMISUPNET.INITIAL_ITER = 12000
    _C.SEMISUPNET.UPDATE_ITER = 1

    # dataloader
    # supervision level
    _C.DATALOADER.SUP_PERCENT = 100.0  # 5 = 5% dataset as labeled set
    _C.DATALOADER.RANDOM_DATA_SEED = 0  # random seed to read data
    _C.DATALOADER.RANDOM_DATA_SEED_PATH = "dataseed/COCO_supervision.txt"

    _C.EMAMODEL = CN()
    _C.EMAMODEL.SUP_CONSIST = True

    _C.DS_OUTPUT_DIR= '/home/shu3090/wcw/clipart_IDCC/clipart_0.2grl_0.9996_3unsup_1sup_10000_ini_0.02pseudo_compare/model_best_IDCC_51.13.pth'
    _C.DP_OUTPUT_DIR= '/home/shu3090/wcw/clipart_/dp_1/model_best.pth'
