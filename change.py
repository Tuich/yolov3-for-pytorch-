import numpy as np
import argparse
import time
import cv2
import os

pic=cv2.imread('images/jiajia.jpg')
print('type of pic:',type(pic))
print('shape:',pic.shape)
#print(pic)

def shiying(filepath,direpath):
    pathdir = os.listdir(filepath)
    for alldir in pathdir:
        child = os.path.join(filepath, alldir)  # 被裁减图片的路径
        direction = os.path.join(direpath, alldir)  # 裁减后图片的路径
        if os.path.isfile(child):
            image = cv2.imread(child)
            iw,ih=image.size
            scale = min(416.0 / float(iw), 416.0 / float(ih))
            nw = int(iw * scale)
            nh=int(ih*scale)
            newimage = image.resize((nw, nh), image.BICUBIC)