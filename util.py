from GPUtil import showUtilization as gpu_usage
#from numba import cuda
import torch

import os
import cv2

from scenedetect import open_video
from scenedetect import SceneManager
from scenedetect import StatsManager
from scenedetect.detectors import ContentDetector

def free_gpu_cache():
    print("Initial GPU Usage")
    gpu_usage()                             

    torch.cuda.empty_cache()

    #cuda.select_device(0)
    #cuda.close()
    #cuda.select_device(0)

    print("GPU Usage after emptying the cache")
    gpu_usage()

def segment_scenes(video_path, img_output, stats_output=None, content_threshold=25):
    img_output = os.path.join(img_output, "scenes")

    if os.path.exists(img_output) is False:
        os.makedirs(img_output, exist_ok=True)

    # Create our video & scene managers, then add the detector.
    video_stream = open_video(video_path)
    stats_manager = StatsManager()
    scene_manager = SceneManager(stats_manager)
    scene_manager.add_detector(
        ContentDetector(threshold=content_threshold))


    stats_file_path = None
    if stats_output is not None:
        stats_file_path = os.path.join(stats_output, "scene_stats.csv")
        scene_manager.detect_scenes(video=video_stream)
    # Each returned scene is a tuple of the (start, end) timecode.
    scene_list = scene_manager.get_scene_list()

    scene_ranges = []
    fps = video_stream.frame_rate

    for i, scene in enumerate(scene_list):
        print(
            'Scene %2d: Start %s / Frame %d, End %s / Frame %d' % (
            i+1,
            scene[0].get_timecode(), scene[0].get_frames(),
            scene[1].get_timecode(), scene[1].get_frames(),))

        scene_ranges.append({
            "start_frame": scene[0].get_frames(),
            "end_frame": scene[1].get_frames(),
        })
        
        video_stream.seek(scene[0].get_frames())
        cv2.imwrite(os.path.join(img_output, '%s.jpg' % (i)), video_stream.read())

        if i == len(scene_list) - 1:
            video_stream.seek(scene[1].get_frames())
            cv2.imwrite(os.path.join(img_output, '%s.jpg' % (i + 1)), video_stream.read())
    
    if stats_file_path is not None:
        stats_manager.save_to_csv(path=stats_file_path)

    return scene_ranges


if __name__ == '__main__':
    free_gpu_cache()