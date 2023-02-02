import cv2
import numpy as np
import torch

def letterbox(im, new_shape=(640, 640), color=(114, 114, 114), auto=True, scaleFill=False, scaleup=True, stride=32):
    # Resize and pad image while meeting stride-multiple constraints
    shape = im.shape[:2]  # current shape [height, width]
    if isinstance(new_shape, int):
        new_shape = (new_shape, new_shape)

    # Scale ratio (new / old)
    r = min(new_shape[0] / shape[0], new_shape[1] / shape[1])
    if not scaleup:  # only scale down, do not scale up (for better val mAP)
        r = min(r, 1.0)

    # Compute padding
    ratio = r, r  # width, height ratios
    new_unpad = int(round(shape[1] * r)), int(round(shape[0] * r))
    dw, dh = new_shape[1] - new_unpad[0], new_shape[0] - new_unpad[1]  # wh padding
    if auto:  # minimum rectangle
        dw, dh = np.mod(dw, stride), np.mod(dh, stride)  # wh padding
    elif scaleFill:  # stretch
        dw, dh = 0.0, 0.0
        new_unpad = (new_shape[1], new_shape[0])
        ratio = new_shape[1] / shape[1], new_shape[0] / shape[0]  # width, height ratios

    dw /= 2  # divide padding into 2 sides
    dh /= 2

    if shape[::-1] != new_unpad:  # resize
        im = cv2.resize(im, new_unpad, interpolation=cv2.INTER_LINEAR)
    top, bottom = int(round(dh - 0.1)), int(round(dh + 0.1))
    left, right = int(round(dw - 0.1)), int(round(dw + 0.1))
    im = cv2.copyMakeBorder(im, top, bottom, left, right, cv2.BORDER_CONSTANT, value=color)  # add border
    return im, ratio, (dw, dh)

def preprocess(img, shape):
    img, _, _ = letterbox(img, shape, auto=False, scaleup=False)
    img = img.transpose((2, 0, 1))[::-1]
    img = np.ascontiguousarray(img)
    return torch.from_numpy(img).unsqueeze(0)

def match_boxes(boxlist1, tracklets, classes):
    matches = []
    for i in range(len(boxlist1)):
        max_iou = 0
        match = -1
        for l,tracklet in enumerate(tracklets):
            iou_value = iou(boxlist1[i][:4], tracklet.tlbr)
            if iou_value > max_iou:
                max_iou = iou_value
                match = l
        matches.append(match)
    out_boxes = []
    for k, (box, c) in enumerate(zip(boxlist1,classes)):
        index = matches[k]
        if index == -1:
            continue
        else:
            out_boxes.append([box, c, tracklets[index].track_id])
        
    return out_boxes


def iou(box1, box2):
    x1_min, y1_min, x1_max, y1_max = box1
    x2_min, y2_min, x2_max, y2_max = box2
    
    # Calculate the (y1, x1, y2, x2) coordinates of the intersection of box1 and box2.
    xi1 = max(x1_min, x2_min)
    yi1 = max(y1_min, y2_min)
    xi2 = min(x1_max, x2_max)
    yi2 = min(y1_max, y2_max)
    inter_area = max(yi2 - yi1, 0) * max(xi2 - xi1, 0)

    # Calculate the Union area by using Formula: Union(A,B) = A + B - Inter(A,B)
    box1_area = (y1_max - y1_min) * (x1_max - x1_min)
    box2_area = (y2_max - y2_min) * (x2_max - x2_min)
    union_area = box1_area + box2_area - inter_area
    
    # Compute the IoU
    iou = inter_area / union_area
    
    return iou
