# limit the number of cpus used by high performance libraries
import json
import os
import sys
import uuid

sys.path.insert(0, './yolov5')

import argparse
from pathlib import Path
import cv2
import torch
import torch.backends.cudnn as cudnn
import tqdm
import tempfile
import time
import numpy as np

from deep_sort.utils.parser import get_config
from deep_sort.deep_sort import DeepSort

import multiprocessing as mp

from detectron2.config import get_cfg
from detectron2.data import MetadataCatalog
from detectron2.utils.logger import setup_logger
from detectron2.utils.video_visualizer import VideoVisualizer
from detectron2.engine.defaults import DefaultPredictor
from detectron2.utils.visualizer import ColorMode
from detectron2.structures.instances import Instances
from detectron2.structures.boxes import Boxes
from yolov5.utils.plots import Annotator, colors

#sys.path.append('./Detic')
sys.path.append('./Detic/third_party/CenterNet2/projects/CenterNet2/')
sys.path.append('./Detic')
from centernet.config import add_centernet_config
from Detic.detic.config import add_detic_config
from Detic.detic.modeling.utils import reset_cls_test
from Detic.detic.predictor import BUILDIN_CLASSIFIER, BUILDIN_METADATA_PATH, get_clip_embeddings


FILE = Path(__file__).resolve()
ROOT = FILE.parents[0]

WINDOW_NAME = "Detic"

def xyxy2xywh(x):
    # Convert nx4 boxes from [x1, y1, x2, y2] to [x, y, w, h] where xy1=top-left, xy2=bottom-right
    y = x.clone() if isinstance(x, torch.Tensor) else np.copy(x)
    y[:, 0] = (x[:, 0] + x[:, 2]) / 2  # x center
    y[:, 1] = (x[:, 1] + x[:, 3]) / 2  # y center
    y[:, 2] = x[:, 2] - x[:, 0]  # width
    y[:, 3] = x[:, 3] - x[:, 1]  # height
    return y


def xywh2xyxy(x):
    # Convert nx4 boxes from [x, y, w, h] to [x1, y1, x2, y2] where xy1=top-left, xy2=bottom-right
    y = x.clone() if isinstance(x, torch.Tensor) else np.copy(x)
    y[:, 0] = x[:, 0] - x[:, 2] / 2  # top left x
    y[:, 1] = x[:, 1] - x[:, 3] / 2  # top left y
    y[:, 2] = x[:, 0] + x[:, 2] / 2  # bottom right x
    y[:, 3] = x[:, 1] + x[:, 3] / 2  # bottom right y
    return y

def fix_detic_directory(detic_dir):
    return os.path.join(os.path.dirname(FILE), 'Detic', detic_dir)

def test_opencv_video_format(codec, file_ext):
    with tempfile.TemporaryDirectory(prefix="video_format_test") as dir:
        filename = os.path.join(dir, "test_file" + file_ext)
        writer = cv2.VideoWriter(
            filename=filename,
            fourcc=cv2.VideoWriter_fourcc(*codec),
            fps=float(30),
            frameSize=(10, 10),
            isColor=True,
        )
        [writer.write(np.zeros((10, 10, 3), np.uint8)) for _ in range(30)]
        writer.release()
        if os.path.isfile(filename):
            return True
        return False

def setup_cfgs(args):
    deepsort_cfg = get_config()
    deepsort_cfg.merge_from_file(args.config_deepsort)
    
    detic_cfg = get_cfg()
    add_centernet_config(detic_cfg)
    add_detic_config(detic_cfg)

    detic_cfg.merge_from_file(ROOT / args.config_detic)

    detic_cfg.merge_from_list(args.opts)
    
    # Set score_threshold for builtin models
    detic_cfg.MODEL.DEVICE = args.device
    detic_cfg.MODEL.RETINANET.SCORE_THRESH_TEST = args.confidence_threshold
    detic_cfg.MODEL.ROI_HEADS.SCORE_THRESH_TEST = args.confidence_threshold
    detic_cfg.MODEL.PANOPTIC_FPN.COMBINE.INSTANCES_CONFIDENCE_THRESH = args.confidence_threshold
    
    detic_cfg.MODEL.ROI_BOX_HEAD.ZEROSHOT_WEIGHT_PATH = 'rand' # load later
    detic_cfg.MODEL.ROI_BOX_HEAD.CAT_FREQ_PATH = fix_detic_directory(detic_cfg.MODEL.ROI_BOX_HEAD.CAT_FREQ_PATH)

    if not args.pred_all_class:
        detic_cfg.MODEL.ROI_HEADS.ONE_CLASS_PER_PROPOSAL = True
    detic_cfg.freeze()
    return deepsort_cfg, detic_cfg

class MOTracker(object):
    def __init__(self, args):
        setup_logger(name="fvcore")
        self.logger = setup_logger()
        self.logger.info("Setup Arguments: " + str(args))

        self.deepsort_cfg, self.detic_cfg = setup_cfgs(args)
        self.deepsort_conf_thresh = args.confidence_threshold

        deepsort_model, device = args.deepsort_model, args.device
        self.deepsort = DeepSort(deepsort_model,
                            max_dist=self.deepsort_cfg.DEEPSORT.MAX_DIST,
                            max_iou_distance=self.deepsort_cfg.DEEPSORT.MAX_IOU_DISTANCE,
                            max_age=self.deepsort_cfg.DEEPSORT.MAX_AGE,
                            n_init=self.deepsort_cfg.DEEPSORT.N_INIT,
                            nn_budget=self.deepsort_cfg.DEEPSORT.NN_BUDGET,
                            use_cuda=(device != 'cpu'))
        
        self.instance_mode = ColorMode.IMAGE
        vocabulary = args.vocabulary
        self.metdata = None
        classifier = None
        if vocabulary == 'custom':
            self.metadata = MetadataCatalog.get('__unused' + str(uuid.uuid4()))
            self.metadata.thing_classes = args.custom_vocabulary.split(',')
            print(self.metadata)
            classifier = get_clip_embeddings(self.metadata.thing_classes)
        else:
            self.metadata = MetadataCatalog.get(
                BUILDIN_METADATA_PATH[vocabulary]
            )
            classifier = fix_detic_directory(BUILDIN_CLASSIFIER[vocabulary])

        num_classes = len(self.metadata.thing_classes)
        self.cpu_device = torch.device("cpu")
        self.detic = DefaultPredictor(self.detic_cfg)
        reset_cls_test(self.detic.model, classifier, num_classes)

    def perform_tracking(self, frames, preds):
        names = self.metadata.get("thing_classes", None)

        # Process detections
        tracked_frames, tracked_preds = [], []
        txt_preds = []
        for frame_idx, (frame, pred) in enumerate(zip(frames, preds)):  # detections per image
            annotator = Annotator(frame, line_width=2, pil=not ascii)
            
            new_pred = Instances(pred.image_size)
            pred_boxes = []
            pred_classes = []
            scores = []

            pred = pred.to(self.cpu_device)
            if pred is not None and len(pred):
                # Print results
                
                xywhs = xyxy2xywh(pred.pred_boxes.tensor)
                confs = pred.scores
                clss = pred.pred_classes

                # pass detections to deepsort
                t4 = time.time()
                outputs = self.deepsort.update(xywhs, confs, clss, frame)
                t5 = time.time()
                self.logger.info(f"DEEP Sort Performed ({frame_idx+1}/{len(frames)}) in {round((t5-t4)*1000)/1000}s")
                # draw boxes for visualization
                if len(outputs) > 0:
                    for j, (output, conf) in enumerate(zip(outputs, confs)):
                        if (conf < self.deepsort_conf_thresh):
                            continue
                        bboxes = output[0:4]
                        id = output[4]
                        cls = output[5]

                        c = int(cls)  # integer class
                        label = f'{id} {names[c]} {conf:.2f}'
                        annotator.box_label(bboxes, label, color=colors(c, True))
                        
                        pred_boxes.append(bboxes)
                        pred_classes.append(c)
                        scores.append(conf)

                        # to MOT format
                        bbox_left = output[0]
                        bbox_top = output[1]
                        bbox_w = output[2] - output[0]
                        bbox_h = output[3] - output[1]
                        # Write MOT compliant results to file
                        txt_preds.append(('%g ' * 10 + '\n') % (frame_idx + 1, id, bbox_left,  # MOT format
                                                            bbox_top, bbox_w, bbox_h, -1, -1, -1, -1))
            else:
                self.deepsort.increment_ages()

            new_pred.pred_boxes = Boxes(torch.tensor(np.array(pred_boxes)))
            new_pred.pred_classes = torch.tensor(np.array(pred_classes))
            new_pred.scores = torch.tensor(np.array(scores))
            
            # frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            # vis_frame = video_visualizer.draw_instance_predictions(frame, new_pred)
            # vis_frame = cv2.cvtColor(vis_frame.get_image(), cv2.COLOR_RGB2BGR)
            vis_frame = cv2.cvtColor(annotator.result(), cv2.COLOR_BGR2RGB)
            tracked_frames.append(vis_frame)
            tracked_preds.append(new_pred)
        
        return tracked_frames, txt_preds

    def _detic_process_video(self, video):
        #video_visualizer = VideoVisualizer(self.metadata, self.instance_mode)

        def extract_predictions(frame, predictions):
            frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            if "instances" in predictions:
                predictions = predictions["instances"].to(self.cpu_device)
            return frame, predictions

        frame_gen = self._frame_from_video(video)
        for frame in frame_gen:
            yield extract_predictions(frame, self.detic(frame))
    
    def _frame_from_video(self, video):
        while video.isOpened():
            success, frame = video.read()
            if success:
                yield frame
            else:
                break

    def process_video(self, video):
        num_frames = int(video.get(cv2.CAP_PROP_FRAME_COUNT))
        
        frames = []
        preds = []
        for frame, pred in tqdm.tqdm(self._detic_process_video(video), total=num_frames):
            frames.append(frame)
            preds.append(pred)
            #if (len(frames) > 30):
            #    break
        video.release()

        frames, preds = self.perform_tracking(frames, preds)
        return frames, preds

def track(tracker, args):

    video = cv2.VideoCapture(args.source)
    width = int(video.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(video.get(cv2.CAP_PROP_FRAME_HEIGHT))
    frames_per_second = video.get(cv2.CAP_PROP_FPS)
    basename = os.path.basename(args.source)
    codec, file_ext = (
        ("x264", ".mkv") if test_opencv_video_format("x264", ".mkv") else ("mp4v", ".mp4")
    )
    frames, preds = tracker.process_video(video)

    if (args.output):
        save_dir = Path(args.output)
        save_dir.mkdir(parents=True, exist_ok=True)

        output_file_vid = os.path.join(args.output, basename)
        output_file_vid = os.path.splitext(output_file_vid)[0] + file_ext
        output_file_txt = os.path.join(args.output, basename)
        output_file_txt = os.path.splitext(output_file_vid)[0] + '.txt'

        if os.path.isfile(output_file_vid):
            os.remove(output_file_vid)
        
        if os.path.isfile(output_file_txt):
            os.remove(output_file_txt)

        output_vid = cv2.VideoWriter(
            filename=output_file_vid,
            # some installation of opencv may not support x264 (due to its license),
            # you can try other format (e.g. MPEG)
            fourcc=cv2.VideoWriter_fourcc(*codec),
            fps=float(frames_per_second),
            frameSize=(width, height),
            isColor=True,
        )
        output_txt = open(output_file_txt, 'w')
        for vis_frame in frames:
            output_vid.write(vis_frame)
        for pred in preds:
            output_txt.write(pred)
        output_txt.close()
        output_vid.release()
    return

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--vocabulary",
        default="lvis",
        choices=['lvis', 'openimages', 'objects365', 'coco', 'custom'],
        help="",
    )
    parser.add_argument(
        "--custom-vocabulary",
        default="",
        help="",
    )
    parser.add_argument("--pred_all_class", action='store_true')
    parser.add_argument(
        "--confidence-threshold",
        type=float,
        default=0.5,
        help="Minimum score for instance predictions to be shown",
    )
    parser.add_argument(
        "--opts",
        help="Modify config options using the command-line 'KEY VALUE' pairs",
        default=[],
        nargs=argparse.REMAINDER,
    )
    parser.add_argument(
        "--config-detic",
        default="Detic/configs/Detic_LCOCOI21k_CLIP_SwinB_896b32_4x_ft4x_max-size.yaml",
        type=str,
        metavar="FILE",
        help="path to config file",
    )

    parser.add_argument('--deepsort-model', type=str, default='osnet_x0_25')
    parser.add_argument("--config-deepsort", type=str, default="deep_sort/configs/deep_sort.yaml")

    parser.add_argument('--source', type=str, default='0', help='source')  # file/folder, 0 for webcam
    parser.add_argument('--output', type=str, default='inference/output', help='output folder')  # output folder
    parser.add_argument('--device', default='', help='cuda device, i.e. 0 or 0,1,2,3 or cpu')
    #parser.add_argument('--save-vid', action='store_true', help='save video tracking results')
    #parser.add_argument('--save-txt', action='store_true', help='save MOT compliant results to *.txt')
    parser.add_argument('--project', default=ROOT / 'runs/track', help='save results to project/name')
    parser.add_argument('--name', default='exp', help='save results to project/name')
    parser.add_argument('--imgsz', '--img', '--img-size', nargs='+', type=int, default=[640], help='inference size h,w')
    
    opt = parser.parse_args()
    opt.imgsz *= 2 if len(opt.imgsz) == 1 else 1  # expand

    tracker = MOTracker(opt)
    with torch.no_grad():
        track(tracker, opt)