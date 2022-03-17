# Copyright (c) OpenMMLab. All rights reserved.
import os
import sys
import os.path as osp
import tempfile
from argparse import ArgumentParser

from pathlib import Path

import cv2
import mmcv
import torch
import json
import numpy

from mmtracking.mmtrack.apis import inference_sot, init_model

sys.path.insert(0, './yolov5')
from yolov5.utils.plots import Annotator, colors

def read_video(video):
    frames = []
    while video.isOpened():
        success, frame = video.read()
        if success:
            frames.append(frame)
        else:
            break
    return frames

def process_result(frame, frame_idx, tlbr, conf):
    # conf = -1 for init-frame-idx == frame-idx
    tlbr = numpy.float64(tlbr)
    conf = numpy.float64(conf)

    img_height, img_width = frame.shape[0], frame.shape[1]

    annotator = Annotator(frame, line_width=2, pil=not ascii)

    tlbr[0] = max(tlbr[0], 0)
    tlbr[1] = max(tlbr[1], 0)

    tlbr[2] = min(tlbr[2], img_width)
    tlbr[3] = min(tlbr[3], img_height)

    tlwh = [tlbr[0], tlbr[1], tlbr[2] - tlbr[0], tlbr[3] - tlbr[1]]
    
    label = f'sot {conf:.2f}'
    annotator.box_label(tlbr, label, color=colors(0, True))

    result_pred_json = {
        "frame": frame_idx,
        "confidence": conf,
        "object_id": -1,
        "class_name": "",
        "top": tlwh[1],
        "left": tlwh[0],
        "height": tlwh[3],
        "width": tlwh[2],
    }

    result_pred_mot = ('%g ' * 10 + '\n') % (frame_idx + 1, -1, tlwh[0],  # MOT format
                                                            tlwh[1], tlwh[2], tlwh[3], conf, -1, -1, -1)

    result_frame = annotator.result()
    return result_frame, result_pred_mot, result_pred_json

def process_video(args, frames, start_frame_idx, end_frame_idx):
    result_frames = []
    result_preds_json = []
    result_preds_mot = []

    # build the model from a config file and a checkpoint file
    model = init_model(args.config_sot, args.checkpoint_sot, device=args.device)

    prog_bar = mmcv.ProgressBar(abs(end_frame_idx - start_frame_idx))

    inc = 1
    if start_frame_idx > end_frame_idx:
        inc = -1

    # test and show/save the images
    for i in range(start_frame_idx, end_frame_idx, inc):
        frame = frames[i]
        if i == start_frame_idx:
            init_bboxes = list(map(float, args.gt_bbox.split(',')))
            init_bboxes[2] += init_bboxes[0]
            init_bboxes[3] += init_bboxes[1]
        frame_idx = abs(i - start_frame_idx)
        result = inference_sot(model, frame, init_bboxes, frame_id=frame_idx)
        
        result_frame, result_pred_mot, result_pred_json = process_result(
            frame, i, result["track_bboxes"][:4], result["track_bboxes"][4]
        )

        result_frames.append(result_frame)
        result_preds_json.append(result_pred_json)
        result_preds_mot.append(result_pred_mot)
        prog_bar.update()

    return result_frames, result_preds_json, result_preds_mot

def track_sot(args):
    video = cv2.VideoCapture(args.source)
    width = int(video.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(video.get(cv2.CAP_PROP_FRAME_HEIGHT))
    frames_per_second = video.get(cv2.CAP_PROP_FPS)
    basename = os.path.basename(args.source)
    codec, file_ext = ("mp4v", ".mp4")

    json_data = {
        "video_metadata": {
            "fps": frames_per_second,
            "width": width,
            "height": height,
            "codec": codec,
            "file_ext": file_ext,
            "original_name": basename,
        },
        "predictions": [],
        "status": "successful"
    }

    init_frame_idx = int(args.init_frame_idx)
    frames = read_video(video)

    print(init_frame_idx)

    if init_frame_idx < 0 or init_frame_idx >= len(frames):
        json_data["status"] = "failed, initial frame is incorrect " + init_frame_idx
        return json_data

    result_frames = []
    result_preds_json = []
    result_preds_mot = []

    if init_frame_idx > 0:
        print("Backwards Processing:")
        cur_frames, cur_preds_json, cur_preds_mot = process_video(
            args, frames, init_frame_idx, -1
        )
        cur_frames = cur_frames[::-1]
        cur_preds_json = cur_preds_json[::-1]
        cur_preds_mot = cur_preds_mot[::-1]
        
        result_frames.extend(cur_frames)
        result_preds_json.extend(cur_preds_json)
        result_preds_mot.extend(cur_preds_mot)
    
    if init_frame_idx < len(frames) - 1:
        print("Forward processing:")
        cur_frames, cur_preds_json, cur_preds_mot = process_video(
            args, frames, init_frame_idx, len(frames)
        )

        if len(result_frames) > 0:
            result_frames.pop()
            result_preds_json.pop()
            result_preds_mot.pop()

        result_frames.extend(cur_frames)
        result_preds_json.extend(cur_preds_json)
        result_preds_mot.extend(cur_preds_mot)
    
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
        output_file_txt = os.path.splitext(output_file_txt)[0] + '.txt'
        
        output_file_json = os.path.join(save_dir, basename)
        output_file_json = os.path.splitext(output_file_json)[0] + '.json'

        init_frame_file = os.path.join(save_dir, basename)
        init_frame_file = os.path.splitext(init_frame_file)[0] + '.jpg'

        if os.path.isfile(output_file_vid):
            os.remove(output_file_vid)
        
        if os.path.isfile(output_file_txt):
            os.remove(output_file_txt)

        if os.path.isfile(init_frame_file):
            os.remove(init_frame_file)

        cv2.imwrite(init_frame_file, result_frames[init_frame_idx])    

        output_vid = cv2.VideoWriter(
            filename=output_file_vid,
            # some installation of opencv may not support x264 (due to its license),
            # you can try other format (e.g. MPEG)
            fourcc=cv2.VideoWriter_fourcc(*codec),
            fps=float(frames_per_second),
            frameSize=(width, height),
            isColor=True,
        )
        for vis_frame in result_frames:
            output_vid.write(vis_frame)
        output_vid.release()

        with open(output_file_txt, 'w') as f:
            for pred in result_preds_mot:
                f.write(pred)

        json_data["predictions"] = result_preds_json

        with open(output_file_json, "w") as f:
            json.dump(json_data, fp=f, indent=4)

    return json_data



if __name__ == '__main__':
    parser = ArgumentParser()
    parser.add_argument('--config-sot', type=str, default='mmtracking/configs/sot/siamese_rpn/siamese_rpn_r50_20e_lasot.py', help='SOT Config file')
    parser.add_argument('--checkpoint-sot', type=str, default=None, help='Checkpoint file')
    parser.add_argument('--source', help='input video file')
    parser.add_argument('--output', default="inference/output", help='output video file (mp4 format)')
    parser.add_argument(
        '--device', default='cuda:0', help='Device used for inference')
    parser.add_argument('--gt-bbox', help='Ground Truth Bounding Box')
    parser.add_argument('--init-frame-idx', default=0, help='Ground Truth Bounding Box')
    args = parser.parse_args()

    with torch.no_grad():
        track_sot(args)
