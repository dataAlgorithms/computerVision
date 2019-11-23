#!/usr/bin/env python
#!coding=utf-8

import re
from collections import defaultdict
import sys
import os
import copy
import cv2

font = cv2.FONT_HERSHEY_SIMPLEX
gMapDict = {"1":"21","2":"24","3":"23","4":"25","5":"26","6":"22"}
gMapDict = {"1":"1","2":"2","3":"3","4":"4","5":"5","6":"6"}

# build result mapping dict
def prtDict1(rtName):

    rtDict = {}
    with open(rtName) as rt:
        for line in rt:
            line = line.strip("\r\n")
            lines = line.split(" ")
            line0 = lines[0]
            img = os.path.basename(line0).split(".")[0]
            linesLeft = lines[1:]
            numby6 = len(linesLeft)/6
            rtDict[img] = []
            for i in range(numby6):
                startIndex = i*6
                endIndex = (i+1)*6
                items = linesLeft[startIndex:endIndex]
                c,t,x,y,w,h = items
                #if img not in rtDict:
                #    rtDict[img] = [[x,y,w,h]]
                #else:
                rtDict[img].append([x, y, w, h, t])

    return rtDict

def prtDict(rtName, end=".jpg"):

    rtDict = {}
    with open(rtName) as rt:
        lines = (line.strip() for line in rt)
        for line in lines:

            img, linesLeft = re.split(end, line)
            img = os.path.basename(img) + end
            if len(linesLeft) == 0:
                rtDict[img] = []
            else:
                linesLeft = linesLeft.strip()
                linesLeft = linesLeft.split(" ")
                numby6 = len(linesLeft)/6
                if img not in rtDict:
                    rtDict[img] = []
                for i in range(numby6):
                    startIndex = i*6
                    endIndex = (i+1)*6
                    items = linesLeft[startIndex:endIndex]
                    c,t,x,y,w,h = items
                    rtDict[img].append([float(x),float(y),float(w),float(h),t])
    return rtDict

# build result mapping dict
def pgtDict1(rtName):

    rtDict = {}
    with open(rtName) as rt:
        for line in rt:
            line = line.strip("\r\n")
            lines = line.split(" ")
            line0 = lines[0]
            img = os.path.basename(line0).split(".")[0]
            linesLeft = lines[1:]
            numby6 = len(linesLeft)/6
            for i in range(numby6):
                startIndex = i*6
                endIndex = (i+1)*6
                items = linesLeft[startIndex:endIndex]
                _,t,x,y,w,h = items
                if img not in rtDict:
                    rtDict[img] = [[float(x),float(y),float(w),float(h), t]]
                else:
                    rtDict[img].append([float(x), float(y), float(w), float(h), t])
    return rtDict


def pgtDict(rtName, end=".jpg"):

    rtDict = {}
    with open(rtName) as rt:
        lines = (line.strip() for line in rt)
        for line in lines:

            img, linesLeft = re.split(end, line)
            img = os.path.basename(img) + end
            if len(linesLeft) == 0:
                rtDict[img] = []
            else:
                linesLeft = linesLeft.strip()
                linesLeft = linesLeft.split(" ")
                numby6 = len(linesLeft)/6
                if img not in rtDict:
                    rtDict[img] = []
                for i in range(numby6):
                    startIndex = i*6
                    endIndex = (i+1)*6
                    items = linesLeft[startIndex:endIndex]
                    c,t,x,y,w,h = items
                    rtDict[img].append([float(x),float(y),float(w),float(h),gMapDict[t]])
    return rtDict
# main
def draw(gt, rt):

    for img in rt:

        im = cv2.imread("images/" + img)
        im_copy = im.copy()
        if img in gt:
            rRois = rt[img]
            gRois = gt[img]

            for rRoi in rRois:
                x, y, w, h, t = rRoi
                cv2.rectangle(im_copy,(int(x),int(y)),(int(x) + int(w), int(y) + int(h)),(0,0,255),1)
                cv2.putText(im_copy, t, (int(x)+int(w), int(y)), font, 0.3, (0, 0, 255), 1)
                #cv2.imshow('draw_rt', im_copy)
                #cv2.waitKey(0)

            for gRoi in gRois:
                x, y, w, h, t = gRoi
                cv2.rectangle(im_copy,(int(x),int(y)),(int(x) + int(w),int(y) + int(h)),(0,255,0),1)
                cv2.putText(im_copy, t, (int(x)+int(w), int(y)+int(h)), font, 0.3, (0, 255, 0), 1)
                #cv2.imshow('draw_gt', im_copy)
                #cv2.waitKey(0)

        cv2.imwrite("draw/" + img, im_copy)
        cv2.destroyAllWindows()

if __name__ == "__main__":
    gtName = sys.argv[1]
    rtName = sys.argv[2]
    gt = pgtDict(gtName)
    rt = prtDict(rtName)
    draw(gt, rt)
