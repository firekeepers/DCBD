import gradio as gr
import os
from PIL import Image
# import your_detection_module  # 假设这是您的检测模型模块
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
        dist_url="tcp://127.0.0.1:50155",
        args=(args,),
    )











# 假设这是您加载检测模型的函数
def load_model():
    model = your_detection_module.load_pretrained_model()
    return model

# 这个函数从文件夹中读取图像，进行检测，并返回原始图像和检测后的图像
def process_images(folder_path):
    images = []
    processed_images = []
    
    # 加载模型
    model = load_model()
    
    # 遍历文件夹，读取图像文件
    for img_file in os.listdir(folder_path):
        if img_file.lower().endswith(('.png', '.jpg', '.jpeg')):
            img_path = os.path.join(folder_path, img_file)
            image = Image.open(img_path)
            images.append(image)
            
            # 进行检测，假设 your_detection_module.detect 返回检测后的图像
            processed_image = your_detection_module.detect(model, image)
            processed_images.append(processed_image)
    
    # 返回原始图像和处理后的图像列表
    return images, processed_images

# 创建 Gradio 界面
iface = gr.Interface(
    fn=process_images,  # 要调用的处理函数
    inputs=gr.inputs.Directory(label="Upload Folder"),  # 输入为文件夹路径
    outputs=[gr.outputs.Image(type="pil", label="Original Image"),
             gr.outputs.Image(type="pil", label="Processed Image")],  # 输出为图像
)

# 启动界面
iface.launch()
