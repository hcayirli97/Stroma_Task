import argparse
import sys
import time

import cv2
import torch

sys.path.append("yolov5/")
sys.path.append("ByteTrack/")

import numpy as np

from get_models import YOLO
from utils_fonks import match_boxes, preprocess
from yolov5.utils.general import non_max_suppression, scale_boxes

def Track(opt):
    input_video_path = opt.video_input_path

    model_path = opt.weights
    device = torch.device(opt.device)
    data_path = opt.data_path
    
    conf_thres, iou_thres = opt.conf_thres, opt.iou_thres
    img_size = [640, 640]


    model, tracker= YOLO(model_path, device, data_path, img_size)

    video_capture = cv2.VideoCapture(input_video_path)

    while True:
        ret, frame = video_capture.read()
        if ret:
            start = time.time()
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
                    cv2.rectangle(frame,(x1,y1),(x2,y2),(98,195,112),1)
                    cv2.putText(frame,"Bolt "+str(track_id),(x1,y1-  10),cv2.FONT_HERSHEY_SIMPLEX,0.5,(0,0,0),4,cv2.LINE_AA)
                    cv2.putText(frame,"Bolt "+str(track_id),(x1,y1-  10),cv2.FONT_HERSHEY_SIMPLEX,0.5,(255,255,255),1,cv2.LINE_AA)
                else:
                    cv2.rectangle(frame,(x1,y1),(x2,y2),(204,51,99),1)
                    cv2.putText(frame,"Nut "+str(track_id),(x1,y1 - 10),cv2.FONT_HERSHEY_SIMPLEX,0.5,(0,0,0),4,cv2.LINE_AA)
                    cv2.putText(frame,"Nut "+str(track_id),(x1,y1 - 10),cv2.FONT_HERSHEY_SIMPLEX,0.5,(255,255,255),1,cv2.LINE_AA)
            
            inference_time = time.time() - start
            cv2.putText(frame, 'FPS: {:.2f}'.format(1/inference_time), (25,25), 
                        cv2.FONT_HERSHEY_SIMPLEX, 
                        1, (255,255,255), 2, cv2.LINE_AA)
            cv2.imshow("frame",frame)
            k = cv2.waitKey(1) & 0xFF
            if k == 27:
                cv2.destroyAllWindows()
                break
        else:
            cv2.destroyAllWindows()
            break


if __name__ == '__main__':

    parser = argparse.ArgumentParser()
    parser.add_argument('--weights', type=str, default='trained_model/best.pt', help='initial weights path')
    parser.add_argument('--data_path', type=str, default='yolov5/data/coco128.yaml', help='model data path')
    parser.add_argument('--device', type=str, default='cuda', help='cuda or cpu')
    parser.add_argument('--conf_thres', type=float, default= 0.7, help='confidence threshold')
    parser.add_argument('--iou_thres', type=float, default= 0.45, help='iou threshold')
    parser.add_argument('--video_input_path', type=str, default='challenge/images/test/test.mp4', help='video path')
    opt = parser.parse_args()

    Track(opt)

