#!/usr/bin/env python
#!coding=utf-8

import numpy as np
import cv2
from collections import defaultdict
import sys
import os
import copy
font = cv2.FONT_HERSHEY_SIMPLEX

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
            lines = line.split(" ")
            line0 = lines[0]
            img = line0
            #img = os.path.basename(line0).split(".")[0]
            #img = "/mnt/cephfs/testData/face_detect.nnie/face_monitor/" + line0.split("/")[-2] + "/" + line0.split("/")[-1]
            linesLeft = lines[1:]
            rtDict[img] = []
            numby6 = len(linesLeft)/4
            for i in range(numby6):
                startIndex = i*4
                endIndex = (i+1)*4
                items = linesLeft[startIndex:endIndex]
                x,y,w,h = items
                rtDict[img].append([x,y,w,h])
                #print('img:', img)
                #raw_input()
    return rtDict

def prtDict(rtName):

    rtDict = {}
    with open(rtName) as rt:
        for line in rt:
            line = line.strip("\r\n")
            lines = line.split(".jpg ")
            line0 = lines[0] + '.jpg'
            #img = os.path.basename(line0).split(".")[0]
            img = line0
            linesLeft = lines[1].split(" ")
            rtDict[img] = []
            numby6 = len(linesLeft)/8
            for i in range(numby6):
                startIndex = i*8
                endIndex = (i+1)*8
                items = linesLeft[startIndex:endIndex]
                t,c,x,y,w,h,r,q = items
                if float(c) < 0.4:
                    continue
                rtDict[img].append([x,y,w,h,c])
    return rtDict

# main
def draw(gt, rt1):

    faceNum = 0
    errNum = 0
    lostNum = 0

    for img in gt:

        gRois = gt[img]
        faceNum += len(gRois)
           
        try:
            rRois = rt1[img]
            gRois = gt[img]
        except:
            for gRoi in gRois:
                x, y, w, h = gRoi
                cv2.rectangle(im_copy,(int(x),int(y)),(int(w)+int(x),int(h)+int(y)),(0,255,0),1)
                lostNum +=  1
                cv2.imwrite("gt_diff_vg/lost." + img.split("/")[-2] +"." + os.path.basename(img), im_copy)  
            continue

        if len(rRois) == 0 and len(gRois) == 0:
            continue

        im = cv2.imread(img)
        if im is None:
            continue
        im_copy = im.copy()
        drawRois = []
        flag = 0
        for rRoi in rRois:
            rx, ry, rw, rh, rc = rRoi

            index = 0
            for gRoi in gRois:
                gx, gy, gw, gh = gRoi
                io = iou([int(rx),int(ry),int(rw),int(rh)],[int(gx),int(gy),int(gw),int(gh)])
                if io >= 0.5:
                    flag = 1
                    del gRois[index]
                    break
                index += 1

            if flag == 0:
                drawRois.append([rx,ry,rw,rh, rc])
    
        errNum += len(drawRois)
        oFlag = 0    
        for oRoi in drawRois:
            x, y, w, h, c = oRoi
            cv2.rectangle(im_copy,(int(x),int(y)),(int(w)+int(x),int(h)+int(y)),(255,0,0),1)
            cv2.putText(im_copy, str(round(float(c),2)), (int(x), int(y)+20), font, 1, (255, 0, 0), 1) 
            oFlag = 1
        if oFlag == 1:
            #cv2.imwrite("diff/error." + img.split("/")[-2] +"." + os.path.basename(img), im_copy)
            pass

        lostNum += len(gRois)
        gFlag = 0
        for gRoi in gRois:
             x, y, w, h = gRoi
             cv2.rectangle(im_copy,(int(x),int(y)),(int(w)+int(x),int(h)+int(y)),(0,255,0),1)
             gFlag = 1
        if gFlag == 1:
            #cv2.imwrite("gt_diff_vg/lost." + img.split("/")[-2] +"." + os.path.basename(img), im_copy)
            pass

    print('faceNum:', faceNum)
    print('errNum/faceNum:', 1.0*errNum/faceNum)
    print('lostNum/faceNum:', 1.0*lostNum/faceNum)

if __name__ == "__main__":
    gt = pgtDict(sys.argv[1])
    rt1 = prtDict(sys.argv[2])
    draw(gt, rt1)
