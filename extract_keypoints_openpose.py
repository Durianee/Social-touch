import json
import os
import sys
import cv2
from sys import platform
import argparse
import numpy as np

import pandas as pd

dir_path = os.path.dirname(os.path.realpath(__file__))
try:
    # Windows Import
    if platform == "win32":
        # Change these variables to point to the correct folder (Release/x64 etc.)
        sys.path.append("D:/Desktop/openpose_GPU/bin");
        os.environ['PATH']  = os.environ['PATH'] + ';' + "D:/Desktop/openpose_GPU/bin;" +  "D:/Desktop/openpose_GPU/bin;"
        import pyopenpose as op
    else:
        # Change these variables to point to the correct folder (Release/x64 etc.)
        sys.path.append('../../python');
        # If you run `make install` (default path is `/usr/local/python` for Ubuntu), you can also access the OpenPose/python module from there. This will install OpenPose and the python library at your desired installation path. Ensure that this is in your python path in order to use it.
        # sys.path.append('/usr/local/python')
        from openpose import pyopenpose as op
except ImportError as e:
    print('Error: OpenPose library could not be found. Did you enable `BUILD_PYTHON` in CMake and have this Python script in the right folder?')
    raise e

import pyopenpose as op

print("成功引入pyopenpose")

parser = argparse.ArgumentParser()

# Custom Params (refer to include/openpose/flags.hpp for more parameters)
params = dict()

params["model_folder"] = "models/"
params["net_resolution"] = "368x256" 
params["face"] = True
params["hand"] = True
# Starting OpenPose
opWrapper = op.WrapperPython()
opWrapper.configure(params)
opWrapper.start()
datum = op.Datum()
# 测试的路径要改一下，自己修改
for video in os.listdir("socialtouch_datascience/2"):
    video_path = os.path.join("socialtouch_datascience/2", video)
    print(f"正在处理视频：{video_path}")
    cap = cv2.VideoCapture(video_path)
    # 获取原视频的尺寸
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = int(cap.get(cv2.CAP_PROP_FPS))
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    # 保存.mp4视频
    fourcc = cv2.VideoWriter_fourcc(*'XVID')
    # 创建VideoWriter对象
    video_name = os.path.basename(video_path).split('.')[0]
    out = cv2.VideoWriter(f'save_video/save_{video_name}.mp4', fourcc, 20, (width, height))
    save_keypoints = {}
    frame_index = 0
    while(cap.isOpened()):
        save_keypoint = {
            "body": [],
            "face": [],
            "left_hand": [],
            "right_hand": [],
        }
        ret, frame = cap.read()
        if not ret:
            break
        imageToProcess = frame
        datum.cvInputData = imageToProcess
        opWrapper.emplaceAndPop(op.VectorDatum([datum]))
        frame = datum.cvOutputData
        # # Display Image
        # cv2.imshow("OpenPose 1.7.0 - Tutorial Python API", frame)
        # if cv2.waitKey(1) & 0xFF == ord('q'):
        #     break
        # 保存视频
        out.write(frame)
        if datum.poseKeypoints is not None:
            # 保存关键点数据
            save_keypoint["body"] = datum.poseKeypoints.tolist()
        if datum.faceKeypoints is not None:
            save_keypoint["face"] = datum.faceKeypoints.tolist()
        if datum.handKeypoints is not None:
            save_keypoint["left_hand"] = datum.handKeypoints[0].tolist()  
            # 注意这里，handKeypoints[0]是左手的关键点，handKeypoints[1]是右手的关键点
            save_keypoint["right_hand"] = datum.handKeypoints[1].tolist()
        save_keypoints[frame_index] = save_keypoint
        frame_index += 1
        print(f"处理进度：{frame_index}/{total_frames}")
        
    with open(f"save_keypoints/{video_name}_keypoints_data.json", "w") as f:
        json.dump(save_keypoints, f)
    # 释放资源
    out.release()
    cap.release()
    cv2.destroyAllWindows()


