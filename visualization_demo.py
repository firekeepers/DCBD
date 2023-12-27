#!/usr/bin/env python3
# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved
import os
os.environ['CUDA_VISIBLE_DEVICES']='1'
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
import gradio as gr
WINDOW_NAME = "COCO detections"
import multiprocessing as mp
from PIL import Image

_M_RGB2YUV = [[0.299, 0.587, 0.114], [-0.14713, -0.28886, 0.436], [0.615, -0.51499, -0.10001]]
_M_YUV2RGB = [[1.0, 0.0, 1.13983], [1.0, -0.39465, -0.58060], [1.0, 2.03211, 0.0]]

# https://www.exiv2.org/tags.html
_EXIF_ORIENT = 274  # exif 'Orientation' tag
def _apply_exif_orientation(image):
    """
    Applies the exif orientation correctly.

    This code exists per the bug:
      https://github.com/python-pillow/Pillow/issues/3973
    with the function `ImageOps.exif_transpose`. The Pillow source raises errors with
    various methods, especially `tobytes`

    Function based on:
      https://github.com/wkentaro/labelme/blob/v4.5.4/labelme/utils/image.py#L59
      https://github.com/python-pillow/Pillow/blob/7.1.2/src/PIL/ImageOps.py#L527

    Args:
        image (PIL.Image): a PIL image

    Returns:
        (PIL.Image): the PIL image with exif orientation applied, if applicable
    """
    if not hasattr(image, "getexif"):
        return image

    try:
        exif = image.getexif()
    except Exception:  # https://github.com/facebookresearch/detectron2/issues/1885
        exif = None

    if exif is None:
        return image

    orientation = exif.get(_EXIF_ORIENT)

    method = {
        2: Image.FLIP_LEFT_RIGHT,
        3: Image.ROTATE_180,
        4: Image.FLIP_TOP_BOTTOM,
        5: Image.TRANSPOSE,
        6: Image.ROTATE_270,
        7: Image.TRANSVERSE,
        8: Image.ROTATE_90,
    }.get(orientation)

    if method is not None:
        return image.transpose(method)
    return image


def convert_PIL_to_numpy(image, format):
    """
    Convert PIL image to numpy array of target format.

    Args:
        image (PIL.Image): a PIL image
        format (str): the format of output image

    Returns:
        (np.ndarray): also see `read_image`
    """
    if format is not None:
        # PIL only supports RGB, so convert to RGB and flip channels over below
        conversion_format = format
        if format in ["BGR", "YUV-BT.601"]:
            conversion_format = "RGB"
        image = image.convert(conversion_format)
    image = np.asarray(image)
    # PIL squeezes out the channel dimension for "L", so make it HWC
    if format == "L":
        image = np.expand_dims(image, -1)

    # handle formats not supported by PIL
    elif format == "BGR":
        # flip channels if needed
        image = image[:, :, ::-1]
    elif format == "YUV-BT.601":
        image = image / 255.0
        image = np.dot(image, np.array(_M_RGB2YUV).T)

    return image
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
class SSS_pipeline():
    def __init__(self,args) -> None:
        self.cfg = setup(args)
        self.Trainer = ATeacherTrainer
        self.model = self.Trainer.build_model(self.cfg)
        self.model_teacher = self.Trainer.build_model(self.cfg)
        # ensem_ts_model = EnsembleTSModel(model_teacher, model)
        self.ensem_ts_model = EnsembleTSModel(self.model, self.model_teacher)

        DetectionCheckpointer(
            self.ensem_ts_model, save_dir=self.cfg.OUTPUT_DIR
        ).resume_or_load(self.cfg.MODEL.WEIGHTS, resume=args.resume)
        self.demo = VisualizationDemo(self.cfg)
    def inference(self,image):
        # if self.cfg.SEMISUPNET.Trainer == "ateacher":

        # work around this bug: https://github.com/python-pillow/Pillow/issues/3973
        image1 = _apply_exif_orientation(image)
        img = convert_PIL_to_numpy(image1, "BGR")
        predictions, visualized_output = self.demo.run_on_image(img)  
        if not "instances" in predictions[0]:
            return image
        return [visualized_output.get_image()[:, :, ::-1]]

def setup_seed(seed):
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)
    torch.backends.cudnn.deterministic = True

class SSSUiPipe():
  def __init__(self) -> None:
    # setup_seed(608123)
    # args = default_argument_parser().parse_args()
    self.model_pipe = SSS_pipeline(args)
    self.SSS_image_List = "/home/shu3090/wcw/datasets/sonar_semi/semi"
    self.SSS_list = os.listdir(self.SSS_image_List)
  
  def ui(self):

    examples_list = [
            ["/home/shu3090/wcw/datasets/sonar_semi/semi/69.jpg"],
            ["/home/shu3090/wcw/datasets/sonar_semi/semi/1917.jpg"],
            ["/home/shu3090/wcw/datasets/sonar_semi/semi/0000140.jpg"],
            ["/home/shu3090/wcw/datasets/sonar_semi/semi/0000439.jpg"],
            ["/home/shu3090/wcw/datasets/sonar_semi/semi/0000561.jpg"],
            ["/home/shu3090/wcw/datasets/sonar_semi/semi/0000184.jpg"],
            ["/home/shu3090/wcw/datasets/sonar_semi/semi/1677.jpg"],
            ["/home/shu3090/wcw/datasets/changjiangkou/val/sonargt2.jpg"],
            ["/home/shu3090/wcw/datasets/changjiangkou/val/sonargt6.jpg"],
            ["/home/shu3090/wcw/datasets/changjiangkou/val/sonargt8.jpg"],
            ["/home/shu3090/wcw/datasets/changjiangkou/val/sonargt10.jpg"],

            # 更多示例图像路径...
        ]
    with gr.Blocks() as demo:
      gr.Markdown("## Side-Scan Sonar Automatic Detection Demo")
      with gr.Row():
        with gr.Column():
            input_image = gr.Image(label="选择一张图片", type="pil")
            input_list = gr.List()
            button = gr.Button("运行", variant="primary")
        with gr.Column():
          
          output = gr.Gallery(label="输出")
      
      input_list = [
        input_image
      ]
    
      output_list = [
        output
      ]
      
      button.click(fn=self.model_pipe.inference, inputs=input_list, outputs=output_list)
    #   demo.add_examples(examples)
      gr.Examples(examples=examples_list,inputs=input_list,outputs=output_list)
    demo.queue(max_size=1)
    demo.launch(server_port=8399, server_name="0.0.0.0", debug=False,share=True)

if __name__ == "__main__":
    setup_seed(608123)
    args = default_argument_parser().parse_args()

    # export:
    # PYTHONWARNINGS = 'ignore:semaphore_tracker:UserWarning'
    torch.multiprocessing.set_sharing_strategy('file_system')
    print("Command Line Args:", args)
    ui_pipeline = SSSUiPipe()
    ui_pipeline.ui()

