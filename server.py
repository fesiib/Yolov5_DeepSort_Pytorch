
from copy import copy
import os
from flask import Flask
from flask_cors import CORS
from flask import request, send_file

import argparse
import tempfile
from pathlib import Path
import json
import numpy as np
from track_bytetrack import MOTracker, track_mot
from track_siamese_rpn import track_sot

app = Flask(__name__)
CORS(app, origins = ["http://localhost:3000"])

app.config["UPLOAD_EXTENSIONS"] = [".mp4", ".jpg", ".png", "webm"]

FILE = Path(__file__).resolve()
ROOT = FILE.parents[0]

opt = None

def fail_with(msg):
    return {
        "status": "failed",
        "message": msg,
    }

@app.route("/process_link_sot", methods=["POST"])
def process_link_sot():
    args = copy(opt)

    if 'video_link' not in request.form:
        return json.dumps(fail_with("No Video Link"))
    video_link = request.form["video_link"]

    if 'object_rectangle' not in request.form:
        return json.dumps(fail_with("No Object to Track"))
    
    object_rectangle = request.form["object_rectangle"]

    args.source = video_link
    args.gt_bbox = object_rectangle

    if 'gpu' in request.form and request.form["gpu"] == "true":
        args.device = 'cuda:0'
    else:
        args.device = 'cpu'

    result = track_sot(args)
    
    responseJSON = {
        "request": {
            "args": str(args),
            "video_link": video_link,
        },
        "result": result,
        "status": "success",
    }
    return json.dumps(responseJSON)

@app.route("/process_file_sot", methods=["POST"])
def process_file_sot():
    args = copy(opt)

    if 'video_file' not in request.files:
        return json.dumps(fail_with("No Video File"))
    video_file = request.files["video_file"]

    if 'object_rectangle' not in request.form:
        return json.dumps(fail_with("No Object to Track"))
    object_rectangle = request.form["object_rectangle"]

    with tempfile.TemporaryDirectory() as td:
        temp_path = Path(td) / video_file.filename
        video_file.save(temp_path)
        args.source = str(temp_path)
        args.gt_bbox = object_rectangle

        if 'gpu' in request.form and request.form["gpu"] == "true":
            args.device = 'cuda:0'
        else:
            args.device = 'cpu'

        result = track_sot(args)
        
        responseJSON = {
            "request": {
                "args": str(args),
                "video_name": video_file.filename,
            },
            "result": result,
            "status": "success",
        }
        return json.dumps(responseJSON)

    

@app.route("/process_link_mot", methods=["POST"])
def process_link_mot():
    args = copy(opt)

    if 'video_link' not in request.form:
        return json.dumps(fail_with("No Video Link"))
    video_link = request.form["video_link"]

    args.source = video_link
    if 'labels' in request.form:
        args.vocabulary = "custom"
        args.custom_vocabulary = request.form["labels"]
    if 'gpu' in request.form and request.form["gpu"] == "true":
        args.device = 'cuda:0'
    else:
        args.device = 'cpu'

    tracker = MOTracker(args)
    result = track_mot(tracker, args)
    
    responseJSON = {
        "request": {
            "args": str(args),
            "video_link": video_link,
        },
        "result": result,
        "status": "success",
    }
    return json.dumps(responseJSON)

@app.route("/process_file_mot", methods=["POST"])
def process_file_mot():
    args = copy(opt)

    if 'video_file' not in request.files:
        return json.dumps(fail_with("No Video File"))
    video_file = request.files["video_file"]

    with tempfile.TemporaryDirectory() as td:
        temp_path = Path(td) / video_file.filename
        video_file.save(temp_path)
        args.source = str(temp_path)

        if 'labels' in request.form:
            args.vocabulary = "custom"
            args.custom_vocabulary = request.form["labels"]
        if 'gpu' in request.form and request.form["gpu"] == "true":
            args.device = 'cuda:0'
        else:
            args.device = 'cpu'

        tracker = MOTracker(args)
        result = track_mot(tracker, args)
        
        responseJSON = {
            "request": {
                "args": str(args),
                "video_name": video_file.filename,
            },
            "result": result,
            "status": "success",
        }
        return json.dumps(responseJSON)

def launch_server():

    app.run(host="0.0.0.0", port=7778)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--vocabulary",
        default="lvis",
        choices=['lvis', 'openimages', 'objects365', 'coco', 'custom'],
        help="Vocabulary",
    )
    parser.add_argument(
        "--custom-vocabulary",
        default="",
        help="",
    )
    parser.add_argument(
        "--pred_all_class",
        action='store_true'
    )
    parser.add_argument(
        "--confidence-threshold",
        type=float,
        default=0.5,
        help="Minimum score for instance predictions to be shown",
    )
    parser.add_argument(
        "--opts",
        help="Modify config options using the command-line 'KEY VALUE' pairs",
        default=["MODEL.WEIGHTS", "./models/Detic_LCOCOI21k_CLIP_SwinB_896b32_4x_ft4x_max-size.pth"],
        nargs=argparse.REMAINDER,
    )
    parser.add_argument(
        "--config-detic",
        default="Detic/configs/Detic_LCOCOI21k_CLIP_SwinB_896b32_4x_ft4x_max-size.yaml",
        type=str,
        metavar="FILE",
        help="path to config file",
    )
    parser.add_argument(
        "--track_thresh",
        type=float,
        default=0.6,
        help="tracking confidence threshold"
    )
    parser.add_argument(
        "--track_buffer",
        type=int,
        default=30,
        help="the frames for keep lost tracks"
    )
    parser.add_argument(
        "--match_thresh",
        type=float,
        default=0.9,
        help="matching threshold for tracking"
    )
    parser.add_argument(
        "--min-box-area",
        type=float,
        default=100,
        help='filter out tiny boxes'
    )
    parser.add_argument(
        "--mot20",
        dest="mot20",
        default=False,
        action="store_true",
        help="test mot20."
    )

    parser.add_argument('--source', type=str, default='0', help='source')  # file/folder, 0 for webcam
    parser.add_argument('--output', type=str, default='inference/output', help='output folder prefix')  # output folder
    parser.add_argument('--device', default='cuda:0', help='cuda device, i.e. 0 or 0,1,2,3 or cpu')
    parser.add_argument('--project', default=ROOT / 'runs/track', help='save results to project/name')
    parser.add_argument('--name', default='exp', help='save results to project/name')
    parser.add_argument('--imgsz', '--img', '--img-size', nargs='+', type=int, default=[640], help='inference size h,w')

    ### SOT
    parser.add_argument('--config-sot', type=str, default='mmtracking/configs/sot/siamese_rpn/siamese_rpn_r50_20e_lasot.py', help='SOT Config file')
    parser.add_argument('--checkpoint-sot', type=str, default=None, help='Checkpoint file')
    parser.add_argument('--gt_bbox', help='Ground Truth Bounding Box')
    
    opt = parser.parse_args()
    opt.imgsz *= 2 if len(opt.imgsz) == 1 else 1  # expand

    launch_server()
    
# --opts MODEL.WEIGHTS ./models/Detic_LCOCOI21k_CLIP_SwinB_896b32_4x_ft4x_max-size.pth   