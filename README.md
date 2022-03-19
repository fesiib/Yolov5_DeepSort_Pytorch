## How to run?

### Server:
```
conda activate mot

python server.py
```

### SOT: Siamese RPN:
```
python track_siamese_rpn.py --source ./dataset/demo_video/demo-2.mp4 --device cuda:0 --gt-bbox 0,0,150,50 --init-frame-idx 200
```

### MOT: Detic + ByteTrack:
```
CUDA_LAUNCH_BLOCKING=1 python track_bytetrack.py --confidence-threshold 0.2 --device cuda:0  --vocabulary lvis --source ./dataset/demo_video/demo-2.mp4 --opts MODEL.WEIGHTS ./models/Detic_LCOCOI21k_CLIP_SwinB_896b32_4x_ft4x_max-size.pth
```

<hr/>

### MOT: Detic + DeepSort:
```
CUDA_LAUNCH_BLOCKING=1 python track_detic.py --confidence-threshold 0.2 --device cuda:0  --vocabulary lvis  --source ./dataset/demo_video/demo-1.mp4 --opts MODEL.WEIGHTS ./models/Detic_LCOCOI21k_CLIP_SwinB_896b32_4x_ft4x_max-size.pth
```
Custom Vocabulary does not seem to work because of META_ARCH issues.


<hr/>

### MOT: YOLOv5 + DeepSort:
```
python track_yolo.py --source ./dataset/demo_video/demo-1.mp4 --yolo_model ./models/yolov5m.pt
```

<hr/>

### Pip Installations
```
pip install mmcv-full -f https://download.openmmlab.com/mmcv/dist/cu113/torch1.10.0/index.html
```