# Stroma Job Interview Task
 
The purpose of this project is to locate and track down bolts and nuts falling down in video footage. It is aimed that the developed algorithm and models work fast and successfully on edge device.

In this study, object detection model and multi-object tracking by detection method were used to solve the problem. The [YOLOv5](https://github.com/ultralytics/yolov5) model was preferred as the object detection model. Although YOLOv5 does not have any articles, it has become one of the most popular object detection models today. The reason for choosing the YOLOv5 model in this project is its easy implementation, low inference time and high performance. [ByteTrack](https://github.com/ifzhang/ByteTrack) algorithm was used as a multi-object tracking by detection method. ByteTrack is a fast and high performance tracking by detection algorithm. Compared to other object tracking algorithms, it can work very fast without using a GPU.

Together with YOLOv5 and ByteTrack, a pipeline was created that provides the targeted high performance and low inference time. It is presented in ONNX and TensorRT format models to maintain high performance on edge devices.

## Getting Started


