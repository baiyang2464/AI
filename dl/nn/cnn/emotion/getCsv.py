# coding:utf8

import cv2
import os
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import csv

def detect(img, cascade):
    rects = cascade.detectMultiScale(img, scaleFactor=1.3, minNeighbors=4, minSize=(30, 30),flags=cv2.CASCADE_SCALE_IMAGE)
    if len(rects) == 0:
        return []
    rects[:, 2:] += rects[:, :2]
    return rects

cascade = cv2.CascadeClassifier("./classifier/haarcascade_frontalface_alt.xml")

f = "./jaffe/"
fs = os.listdir(f)
data = np.zeros([213, 48*48], dtype=np.uint8)
label = np.zeros([213], dtype=int)
i = 0
for f1 in fs:
    tmp_path = os.path.join(f, f1)
    if not os.path.isdir(tmp_path):
        print(tmp_path[len(f):])
        img = cv2.imread(tmp_path, 1)
        dst = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        rects = detect(dst, cascade)
        for x1, y1, x2, y2 in rects:
            cv2.rectangle(img,(x1+10,y1+20),(x2-10,y2),(0,255,255),2)
            # 调整截取脸部区域大小
            img_roi = np.uint8([y2-(y1+20), (x2-10)-(x1+10)])
            roi = dst[y1+20:y2, x1+10:x2-10]
            img_roi = roi
            re_roi = cv2.resize(img_roi, (48,48))
            # 获得表情label
            img_label = tmp_path[len(f)+3:len(f)+5]
            # print(img_label)
            if img_label == 'AN':
                label[i] = 0
            elif img_label == 'DI':
                label[i] = 1
            elif img_label == 'FE':
                label[i] = 2
            elif img_label == 'HA':
                label[i] = 3
            elif img_label == 'SA':
                label[i] = 4
            elif img_label == 'SU':
                label[i] = 5
            elif img_label == 'NE':
                label[i] = 6
            else:
                print("get label error.......\n")

            data[i][0:48*48] = np.ndarray.flatten(re_roi)
            i = i + 1

            # cv2.imshow("src", dst)
            # cv2.imshow("img", img)
            # if cv2.waitKey() == 32:
            #     continue

with open(r"./face.csv","w") as csvfile:
    writer = csv.writer(csvfile)
    writer.writerow(['emotion', 'pixels'])
    for i in range(len(label)):
        data_list = list(data[i])
        b = " ".join(str(x) for x in data_list)
        l = np.hstack([label[i], b])
        writer.writerow(l)
