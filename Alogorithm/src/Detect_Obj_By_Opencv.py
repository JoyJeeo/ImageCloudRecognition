import cv2
import torch
import urllib.request
import numpy as np
from models.common import DetectMultiBackend
from utils.torch_utils import select_device
from utils.augmentations import letterbox
from utils.general import non_max_suppression, scale_boxes
url = "http://192.168.4.1:80/capture" # esp32-cam的ip地址
device = select_device("")
model = DetectMultiBackend("yolov5s.pt", device=device, dnn=False, data="", fp16=False)
while True:
    img_resp = urllib.request.urlopen(url, timeout=10) # 从url获取图像数据
    imgnp = np.array(bytearray(img_resp.read()), dtype=np.uint8)
    frame = cv2.imdecode(imgnp, -1) # 解码jpeg图像数据
    # 预处理
    processed_img, _, _ = letterbox(frame, 640)
    processed_img = processed_img[:, :, ::-1].transpose(2, 0, 1)
    processed_img = np.ascontiguousarray(processed_img) / 255.0
    tensor = torch.from_numpy(processed_img).to(device).float()[None]
    # 推理
    pred = model(tensor)
    pred = non_max_suppression(pred)[0]
    pred[:, :4] = scale_boxes(processed_img.shap[1:], pred[:, :4], frame.shape).round() # 坐标还原
    # 结果使用
    if pred is not None:
        for det in pred:
            if det[5] == 0: # 人设置框为绿色
                x1,y1,x2,y2 = map(int, det[:4])
                print(x1,y1,x2,y2) # 输出
                cv2.rectangle(frame, (x1,y1),(x2,y2),(0,255,0),2)
            if det[5] == 2: # 汽车设置为红色
                x1,y1,x2,y2 = map(int, det[:4])
                print(x1,y1,x2,y2) # 输出 
                cv2.rectangle(frame, (x1,y1),(x2,y2),(0,255,0),2)
    # 显示带有预测结果的帧
    cv2.imshow("TOLOv5 Detection", frame)
    cv2.waitKey(1)
# 释放摄像头资源并关闭所有窗口
cv2.destoryAllWindows()