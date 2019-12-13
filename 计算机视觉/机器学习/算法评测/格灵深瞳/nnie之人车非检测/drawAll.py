#!/usr/bin/env python
#!coding=utf-8

import numpy as np
import cv2
from collections import defaultdict
import sys
import os
import copy
font = cv2.FONT_HERSHEY_SIMPLEX

nameMap  = {"person":2, "car":1, "bicycle":3, "tricycle":4}

# calculate iou
def iou(b1, b2):
    iou_val = 0.0
    x1 = np.max([b1[0], b2[0]])
    y1 = np.max([b1[1], b2[1]])
    x2 = np.min([b1[0] + b1[2], b2[0] + b2[2]])
    y2 = np.min([b1[1] + b1[3], b2[1] + b2[3]])
    w = np.max([0, x2 - x1])
    h = np.max([0, y2 - y1])
    if w != 0 and h != 0:
        iou_val = float(w * h) / (b1[2] * b1[3] + b2[2] * b2[3] - w * h)
    return iou_val

def pgtDict(rtName):

    rtDict = {}
    with open(rtName) as rt:
        for line in rt:
            line = line.strip("\r\n")
            try:
                img, linesLeft = line.split(".jpg,")
            except:
                # no detect info
                rtDict[img] = []
                continue
            img = img + '.jpg'
            #img = os.path.basename(img)
            linesLeft = linesLeft.split(",")
            if img not in rtDict:
                rtDict[img] = []           
            numby5 = len(linesLeft)/5
            for i in range(numby5):
                startIndex = i*5
                endIndex = (i+1)*5
                items = linesLeft[startIndex:endIndex]
                t,x,y,w,h = items
                rtDict[img].append([x,y,w,h,t])

    return rtDict

def prtDict(rtName):

    rtDict = {}
    with open(rtName) as rt:
        for line in rt:
            line = line.strip("\r\n")
            lines = line.split(" ")
            line0 = lines[0]
            img = line0
            #img = os.path.basename(img)
            #img = os.path.basename(line0).split(".")[0]
            linesLeft = lines[1:]
            if img not in rtDict:
                rtDict[img] = []
            numby8 = len(linesLeft)/5
            for i in range(numby8):
                startIndex = i*5
                endIndex = (i+1)*5
                items = linesLeft[startIndex:endIndex]
                c,x,y,w,h = items

                x = int(float(x))
                y = int(float(y))
                w = int(float(w))
                h = int(float(h))
                if float(c) < 0.5:
                    continue
                rtDict[img].append([x,y,w,h,c])

    return rtDict

# main
def draw(gt, rt, cType=2, virtual=0):

    errorNums = 0
    lostNums = 0
    totalNums = 0

    for img in gt:

        im = cv2.imread(img)
        im_copy = im.copy()
        gRois = gt[img]
        if len(gRois) == 0:
            continue
        for gRoi in gRois:
            gx, gy, gw, gh, gts = gRoi

        try:
            rRois = rt[img]
            gRois = gt[img]
        except:
            lostNums += 1

            for gRoi in gRois:
                x, y, w, h, t = gRoi
                cv2.rectangle(im_copy,(int(x),int(y)),(int(w)+int(x),int(h)+int(y)),(0,255,0),1)
            cv2.imwrite("drawAll/draw." + os.path.basename(img), im_copy)
            continue

        for gRoi in gRois:
            x, y, w, h, t = gRoi
            cv2.rectangle(im_copy,(int(x),int(y)),(int(w)+int(x),int(h)+int(y)),(0,255,0),1)

        for rRoi in rRois:
            x, y, w, h, t = rRoi
            cv2.rectangle(im_copy,(int(x),int(y)),(int(w),int(h)),(255,0,0),1)

        cv2.imwrite("drawAll/draw." + os.path.basename(img), im_copy)

if __name__ == "__main__":
    gt = pgtDict(sys.argv[1])
    rt = prtDict(sys.argv[2])
    draw(gt, rt)
