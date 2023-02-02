import torch
import sys
sys.path.append("yolov5/")
from yolov5.models.common import DetectMultiBackend
from ByteTrack.yolox.tracker.byte_tracker import BYTETracker

class TrackArgs:
    track_thresh = 0.5
    track_buffer = 15
    match_thresh = 0.7
    aspect_ratio_thresh = 10.0
    min_box_area = 1.0
    mot20 = False

def YOLO(weight_path, device, data, imgsz):
    Model = DetectMultiBackend(weight_path, device=device, dnn=False, data=data, fp16=False)
    Model.eval()
    Model.warmup(imgsz=(1 if Model.pt else 1, 3, *imgsz))
    args = TrackArgs()
    tracker = BYTETracker(args , frame_rate=5)
    return Model, tracker