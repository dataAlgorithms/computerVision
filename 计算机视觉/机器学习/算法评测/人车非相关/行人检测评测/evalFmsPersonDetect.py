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
            lines = line.split(",")
            line0 = lines[0]
            img = os.path.basename(line0)
            try:
                linesLeft = lines[1:]
                if img not in rtDict:
                    rtDict[img] = []
            except:
                rtDict[img] = []
                continue
            numby5 = len(linesLeft)/5
            for i in range(numby5):
                startIndex = i*5
                endIndex = (i+1)*5
                items = linesLeft[startIndex:endIndex]
                t,x,y,w,h = items
                rtDict[img].append([int(x),int(y),int(w),int(h),int(t)])
    return rtDict

def prtDict(rtName):

    rtDict = {}
    with open(rtName) as rt:
        for line in rt:
            line = line.strip("\r\n")
            if '.jpg ' in line:
                lines = line.split(".jpg ")
            else:
                lines = line.split(".jpg")
            img = os.path.basename(lines[0]) + '.jpg'
            try:
                linesLeft = lines[1].split(" ")
                if img not in rtDict:
                    rtDict[img] = []
            except:
                rtDict[img] = []
                continue
            numby6 = len(linesLeft)/5
            for i in range(numby6):
                startIndex = i*5
                endIndex = (i+1)*5
                items = linesLeft[startIndex:endIndex]
                c,x,y,w,h = items
                #if float(c) < 0.5:
                #    continue
                x = float(x)
                y = float(y)
                w = float(w) - float(x)
                h = float(h) - float(y)
                rtDict[img].append([int(x),int(y),int(w),int(h),2])

    return rtDict

# main
def drawOld(gt, rt, cType):

    errorNums = 0
    lostNums = 0
    totalNums = 0

    for img in gt:

        gRois = gt[img]
        for gRoi in gRois:
            gx, gy, gw, gh, gts = gRoi
            if gts == cType:
                totalNums += 1

        #print('img:', img)
        try:
            rRois = rt[img]
            gRois = gt[img]
        except:
            
            lostNums += 1
            continue

        drawRois = []

        for rRoi in rRois:
            rx, ry, rw, rh, rts = rRoi

            flag = 0
            index = 0
            for gRoi in gRois:
                gx, gy, gw, gh, gts = gRoi
                io = iou([int(rx),int(ry),int(rw),int(rh)],[int(gx),int(gy),int(gw),int(gh)])
                if io >= 0.5 and rts == gts and gts == cType:
                    flag = 1
                    del gRois[index]
                    break
                index += 1

            if flag == 0:
                drawRois.append([rx,ry,rw,rh,rts])

        # error detect 
        errorNum = 0
        for oRoi in drawRois:
            x, y, w, h, t = oRoi
            if t != cType:
                continue
            errorNum += 1
        errorNums += errorNum
 
        # lost detect
        lostNum = 0
        for gRoi in gRois:
            x, y, w, h, t = gRoi
            if t != cType:
                continue
            lostNum += 1
        lostNums += lostNum

        #print('totalNums:', totalNums)
        #print('lostNums:', lostNums)
        #print('errNums:', errorNums)
        #raw_input()

    print('totalNums:', totalNums)
    print('lostRatioo(lostNums/totalNums):', 1.0*lostNums/totalNums)
    print('errRatio(errorNums/totalNums):', 1.0*errorNums/totalNums)

def draw(gt, rt, cType, virtual=1):

    errorNums = 0
    lostNums = 0
    totalNums = 0

    for img in gt:

        gRois = gt[img]
        if len(gRois) == 0:
            continue
        for gRoi in gRois:
            gx, gy, gw, gh, gts = gRoi
            if gts == cType:
                totalNums += 1

        if int(virtual) == 1:
            im = cv2.imread("/mnt/cephfs/testData/fms/personDetect/" + img)
            im_copy = im.copy()

        #print('img:', img)
        try:
            rRois = rt[img]
            gRois = gt[img]
        except:
            for gRoi in gRois:
                gx, gy, gw, gh, gts = gRoi
                if gts == cType:
                    lostNums += 1

                if int(virtual) == 1:
                    cv2.rectangle(im_copy,(int(gx),int(gy)),(int(gw)+int(gx),int(gh)+int(gy)),(0,255,0),1)
                    cv2.putText(im_copy, str(gts), (int(gx), int(gy)+20), font, 1, (0, 255, 0), 1)

            cv2.imwrite("diff/draw." + os.path.basename(img), im_copy)
            continue

        #print('rRois:', rRois)
        #print('gRois:', gRois)
        drawRois = []

        for rRoi in rRois:
            rx, ry, rw, rh, rts = rRoi

            flag = 0
            index = 0
            for gRoi in gRois:
                gx, gy, gw, gh, gts = gRoi
                io = iou([int(rx),int(ry),int(rw),int(rh)],[int(gx),int(gy),int(gw),int(gh)])
                if io >= 0.5 and rts == gts and gts == cType:
                    flag = 1
                    del gRois[index]
                    break
                index += 1

            if flag == 0:
                drawRois.append([rx,ry,rw,rh,rts])

        # error detect 
        eflag = 0
        errorNum = 0
        for oRoi in drawRois:
            x, y, w, h, t = oRoi
            if t != cType:
                continue
            errorNum += 1
            if int(virtual) == 1:
                cv2.rectangle(im_copy,(int(x),int(y)),(int(w)+int(x),int(h)+int(y)),(255,0,0),1)
                cv2.putText(im_copy, str(t), (int(x), int(y)+20), font, 1, (255, 0, 0), 1)
            eflag = 1
        errorNums += errorNum
 
        # lost detect
        lostNum = 0
        lflag = 0
        for gRoi in gRois:
            x, y, w, h, t = gRoi
            if t != cType:
                continue
            lostNum += 1
            if int(virtual) == 1:
                cv2.rectangle(im_copy,(int(x),int(y)),(int(w)+int(x),int(h)+int(y)),(0,255,0),1)
            lflag = 1
        lostNums += lostNum

        if int(virtual) == 1:
            if eflag == 1 or lflag == 1:
                cv2.imwrite("diff/draw." + os.path.basename(img), im_copy)

        #print('totalNums:', totalNums)
        #print('lostNums:', lostNums)
        #print('errNums:', errorNums)
        #raw_input()

    print('totalNums:', totalNums)
    print('lostRatioo(lostNums/totalNums):', 1.0*lostNums/totalNums)
    print('errRatio(errorNums/totalNums):', 1.0*errorNums/totalNums)

if __name__ == "__main__":
    gt = pgtDict(sys.argv[1])
    rt = prtDict(sys.argv[2])
    ctype = int(sys.argv[3])
    draw(gt, rt, ctype)
