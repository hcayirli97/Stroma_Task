import torch 
import cv2
import sys
sys.path.append("yolov5/")
sys.path.append("ByteTrack/")

from utils_fonks import preprocess, match_boxes
from get_models import YOLO
import numpy as np

from yolov5.utils.general import non_max_suppression, scale_boxes




input_video_path = "challenge/images/test/test.mp4"

model_path = "yolov5/runs/train/yolov5n_/weights/best.pt"
device = torch.device("cuda")
data = "yolov5/data/coco128.yaml"
img_size = [640, 640]
conf_thres, iou_thres = 0.5, 0.5


model, tracker= YOLO(model_path, device, data, img_size)

video_capture = cv2.VideoCapture(input_video_path)

while True:
    ret, frame = video_capture.read()
    if ret:
        bboxes = []
        classes = []
        im = preprocess(frame, img_size[0])
        im = im.to(device, non_blocking=True).float()
        im /= 255

        pred = model(im, augment=False, visualize=False)
        pred = non_max_suppression(pred, conf_thres, iou_thres, None, False, max_det=50)
        for i, det in enumerate(pred):
            if len(det):

                det[:, :4] = scale_boxes(im.shape[2:], det[:, :4], frame.shape).round()

                for *xyxy, conf, cls in reversed(det):
                    c = int(cls) 
                    x1 , y1 , x2 , y2 = xyxy
                    x1 , y1 , x2 , y2 = int(x1), int(y1), int(x2), int(y2)
                    bboxes.append([x1 , y1 , x2 , y2, float(conf)])
                    classes.append(int(cls))
                    # if c == 0:
                    #     cv2.rectangle(frame,(x1,y1),(x2,y2),(255,0,0),1)
                    # else:
                    #     cv2.rectangle(frame,(x1,y1),(x2,y2),(0,255,0),1)

        try:
            input_track = torch.tensor(bboxes, dtype=torch.float32)
            tracklets = tracker.update(input_track, img_size, img_size)
        except:
            input_track = torch.empty(1,6)
            tracklets = tracker.update(input_track , img_size , img_size)

        out_boxes = match_boxes(bboxes, tracklets, classes)

        for box in out_boxes:
            x1 ,y1,x2, y2 = box[0][:4]
            c = box[1]
            track_id = box[2]
            if c == 0:
                cv2.rectangle(frame,(x1,y1),(x2,y2),(0,0,255),1)
                cv2.putText(frame,"Bolt "+str(track_id),(x1,y1-  10),cv2.FONT_HERSHEY_SIMPLEX,0.5,(0,0,0),4,cv2.LINE_AA)
                cv2.putText(frame,"Bolt "+str(track_id),(x1,y1-  10),cv2.FONT_HERSHEY_SIMPLEX,0.5,(255,255,255),1,cv2.LINE_AA)
            else:
                cv2.rectangle(frame,(x1,y1),(x2,y2),(255,0,0),1)
                cv2.putText(frame,"Nut "+str(track_id),(x1,y1 - 10),cv2.FONT_HERSHEY_SIMPLEX,0.5,(0,0,0),4,cv2.LINE_AA)
                cv2.putText(frame,"Nut "+str(track_id),(x1,y1 - 10),cv2.FONT_HERSHEY_SIMPLEX,0.5,(255,255,255),1,cv2.LINE_AA)
        
        print("---------------------")
        cv2.imshow("frame",frame)
        cv2.waitKey(1)
    else:
        cv2.destroyAllWindows()
        break