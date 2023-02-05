# Stroma Job Interview Task
 
The purpose of this project is to locate and track down bolts and nuts falling down in video footage. It is aimed that the developed algorithm and models work fast and successfully on edge device.

In this study, object detection model and multi-object tracking by detection method were used to solve the problem. The [YOLOv5](https://github.com/ultralytics/yolov5) model was preferred as the object detection model. Although YOLOv5 does not have any articles, it has become one of the most popular object detection models today. The reason for choosing the YOLOv5 model in this project is its easy implementation, low inference time and high performance. [ByteTrack](https://github.com/ifzhang/ByteTrack) algorithm was used as a multi-object tracking by detection method. ByteTrack is a fast and high performance tracking by detection algorithm. Compared to other object tracking algorithms, it can work very fast without using a GPU.

Together with YOLOv5 and ByteTrack, a pipeline was created that provides the targeted high performance and low inference time. It is presented in ONNX and TensorRT format models to maintain high performance on edge devices.

## Firstly
First of all, the videos in the dataset were extracted with their annotations and saved in YOLO format. The operations are included in the **extract_video.py** file. After the dataset was created, the YOLOv5n model was trained with the pretrained model for 10 epochs. As a result of the training, the results of the test set are shown in the graphics below.

![PR](https://github.com/hcayirli97/Stroma_Task/blob/main/imgs/PR_curve.png)
![CM](https://github.com/hcayirli97/Stroma_Task/blob/main/imgs/confusion_matrix.png)

The resulting Pytorch model is exported as onnx and tensorrt and is located in the input video path [trained_models](https://github.com/hcayirli97/Stroma_Task/tree/main/trained_models) folder.

## Detection and Tracking Pipeline

The Detection and Tracking pipeline is located in the [main.py](https://github.com/hcayirli97/Stroma_Task/blob/main/main.py). It is recorded as a new video after the locations of the objects are detected and tracked by processing the input video. It can be run by giving the video path, the path of the model to be used, the output video recording path, etc. as the input argument.

How the code can be run is shared in the [Stroma_Main_Colab.ipynb](https://github.com/hcayirli97/Stroma_Task/blob/main/Stroma_Main_Colab.ipynb). After the necessary libraries are installed, the result can be obtained by giving the necessary input arguments to the main.py file.
```sh  
python main.py --weights "trained_models/yolov5n_best.engine" \
                --video_input_path "input_video/test.mp4" \
                --save_path "results/"     
```
   
 ## Result
 
 After giving **test.mp4** video as input, model output was recorded as video. In the below, there is a short video clip from the video output.
 
 ![Result](https://github.com/hcayirli97/Stroma_Task/blob/main/imgs/Result.gif)
