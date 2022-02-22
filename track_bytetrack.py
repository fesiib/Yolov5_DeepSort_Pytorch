# limit the number of cpus used by high performance libraries
import json
import os
import sys
import uuid

from ByteTrack.yolox.tracker.byte_tracker import BYTETracker

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
from Detic.detic.predictor import BUILDIN_CLASSIFIER, BUILDIN_METADATA_PATH


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

def fix_detic_directory(detic_dir):
    return os.path.join(os.path.dirname(FILE), 'Detic', detic_dir)

def get_clip_embeddings(vocabulary, prompt='a '):
    from Detic.detic.modeling.text.text_encoder import build_text_encoder
    text_encoder = build_text_encoder(pretrain=True)
    text_encoder.eval()
    texts = [prompt + x for x in vocabulary]
    emb = text_encoder(texts).detach().permute(1, 0).contiguous().cpu()
    return emb

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

def setup_cfg(args):
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
    return detic_cfg

class MOTracker(object):
    def __init__(self, args):
        setup_logger(name="fvcore")
        self.logger = setup_logger()

        self.detic_cfg = setup_cfg(args)

        self.bytetrack = BYTETracker(args, frame_rate=30)
        self.min_box_area = args.min_box_area
        
        self.instance_mode = ColorMode.IMAGE
        vocabulary = args.vocabulary
        self.metdata = None
        classifier = None
        if vocabulary == 'custom':
            self.metadata = MetadataCatalog.get('__unused' + str(uuid.uuid4()))
            self.metadata.thing_classes = args.custom_vocabulary.split(',')
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
        mot_preds = []
        json_preds = []
        for frame_idx, (frame, pred) in enumerate(zip(frames, preds)):  # detections per image
            annotator = Annotator(frame, line_width=2, pil=not ascii)
            
            new_pred = Instances(pred.image_size)
            pred_boxes = []
            pred_classes = []
            scores = []

            pred = pred.to(self.cpu_device)
            if pred is not None and len(pred):
                # Print results

                #xywhs = xyxy2xywh(pred.pred_boxes.tensor)
                xyxy = pred.pred_boxes.tensor
                confs = pred.scores
                clss = pred.pred_classes
                inputs = torch.cat((xyxy, torch.unsqueeze(confs, dim=1)), dim=1)

                info_img = frame.shape[0], frame.shape[1]
                img_size = frame.shape[0], frame.shape[1]

                # pass detections to bytetarack
                t4 = time.time()
                outputs = self.bytetrack.update(inputs, info_img, img_size)
                t5 = time.time()
                self.logger.info(f"DEEP Sort Performed ({frame_idx+1}/{len(frames)}) in {round((t5-t4)*1000)/1000}s")
                # draw boxes for visualization
                if len(outputs) > 0:
                    for j, (output, conf, cls) in enumerate(zip(outputs, confs, clss)):
                        tlbr = output.tlbr

                        tlbr[0] = max(tlbr[0], 0)
                        tlbr[1] = max(tlbr[1], 0)

                        tlbr[2] = min(tlbr[2], img_size[1])
                        tlbr[3] = min(tlbr[3], img_size[0])

                        tlwh = [tlbr[0], tlbr[1], tlbr[2] - tlbr[0], tlbr[3] - tlbr[1]]

                        #vertical = tlhw[2] / tlhw[3] > 1.6
                        if tlwh[2] * tlwh[3] < self.min_box_area:
                            continue
                        
                        tid = output.track_id
                        c = int(cls)  # integer class
                        label = f'{tid} {names[c]} {conf:.2f}'
                        annotator.box_label(tlbr, label, color=colors(c, True))
                        
                        pred_boxes.append(tlbr)
                        pred_classes.append(c)
                        scores.append(conf)

                        # to JSON format
                        json_preds.append({
                            "frame": frame_idx,
                            "confidence": conf, 
                            "object_id": tid,
                            "class_name": names[c],
                            "top": tlwh[1],
                            "left": tlwh[0],
                            "height": tlwh[3],
                            "width": tlwh[2],
                        })

                        # to MOT format
                        # Write MOT compliant results to file
                        mot_preds.append(('%g ' * 10 + '\n') % (frame_idx + 1, tid, tlwh[0],  # MOT format
                                                            tlwh[1], tlwh[2], tlwh[3], conf, -1, -1, -1))

            new_pred.pred_boxes = Boxes(torch.tensor(np.array(pred_boxes)))
            new_pred.pred_classes = torch.tensor(np.array(pred_classes))
            new_pred.scores = torch.tensor(np.array(scores))
            
            # frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            # vis_frame = video_visualizer.draw_instance_predictions(frame, new_pred)
            # vis_frame = cv2.cvtColor(vis_frame.get_image(), cv2.COLOR_RGB2BGR)
            vis_frame = cv2.cvtColor(annotator.result(), cv2.COLOR_BGR2RGB)
            tracked_frames.append(vis_frame)
            tracked_preds.append(new_pred)
        
        return tracked_frames, mot_preds, json_preds

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
            # if (len(frames) > 10):
            #     break
        video.release()

        frames, mot_preds, json_preds  = self.perform_tracking(frames, preds)
        return frames, mot_preds, json_preds

def track_mot(tracker, args):

    video = cv2.VideoCapture(args.source)
    width = int(video.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(video.get(cv2.CAP_PROP_FRAME_HEIGHT))
    frames_per_second = video.get(cv2.CAP_PROP_FPS)
    basename = os.path.basename(args.source)
    codec, file_ext = (
        ("x264", ".mkv") if test_opencv_video_format("x264", ".mkv") else ("mp4v", ".mp4")
    )
    frames, mot_preds, json_preds = tracker.process_video(video)

    if (args.output):
        output_id = 0
        while True:
            cur_output = args.output + str(output_id)
            save_dir = Path(cur_output)
            if save_dir.exists() is False:
                save_dir.mkdir(parents=True, exist_ok=True)
                break
            output_id += 1
        save_dir = args.output + str(output_id)
        output_file_vid = os.path.join(save_dir, basename)
        output_file_vid = os.path.splitext(output_file_vid)[0] + file_ext
        
        output_file_txt = os.path.join(save_dir, basename)
        output_file_txt = os.path.splitext(output_file_vid)[0] + '.txt'
        
        output_file_json = os.path.join(save_dir, basename)
        output_file_json = os.path.splitext(output_file_vid)[0] + '.json'

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
        for vis_frame in frames:
            output_vid.write(vis_frame)
        output_vid.release()

        with open(output_file_txt, 'w') as f:
            for pred in mot_preds:
                f.write(pred)

        json_data = {
            "video_metadata": {
                "fps": frames_per_second,
                "width": width,
                "height": height,
                "codec": codec,
                "file_ext": file_ext,
                "original_name": basename,
            },
            "predictions": json_preds,
        }
        with open(output_file_json, "w") as f:
            json.dump(json_data, fp=f, indent=4)
        return json_data
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

    parser.add_argument("--track_thresh", type=float, default=0.6, help="tracking confidence threshold")
    parser.add_argument("--track_buffer", type=int, default=30, help="the frames for keep lost tracks")
    parser.add_argument("--match_thresh", type=float, default=0.9, help="matching threshold for tracking")
    parser.add_argument("--min-box-area", type=float, default=100, help='filter out tiny boxes')
    parser.add_argument("--mot20", dest="mot20", default=False, action="store_true", help="test mot20.")

    parser.add_argument('--source', type=str, default='0', help='source')  # file/folder, 0 for webcam
    parser.add_argument('--output', type=str, default='inference/output', help='output folder prefix')  # output folder
    parser.add_argument('--device', default='', help='cuda device, i.e. 0 or 0,1,2,3 or cpu')
    #parser.add_argument('--save-vid', action='store_true', help='save video tracking results')
    #parser.add_argument('--save-txt', action='store_true', help='save MOT compliant results to *.txt')
    parser.add_argument('--project', default=ROOT / 'runs/track', help='save results to project/name')
    parser.add_argument('--name', default='exp', help='save results to project/name')
    parser.add_argument('--imgsz', '--img', '--img-size', nargs='+', type=int, default=[640], help='inference size h,w')
    
    opt = parser.parse_args()
    opt.imgsz *= 2 if len(opt.imgsz) == 1 else 1  # expand

    with torch.no_grad():
        tracker = MOTracker(opt)
        track_mot(tracker, opt)