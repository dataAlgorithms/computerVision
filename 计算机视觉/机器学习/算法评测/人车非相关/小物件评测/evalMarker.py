#!/usr/bin/env python
#!coding=utf-8

import re
import numpy as np
import cv2
from collections import defaultdict
import sys
import os
import copy
from collections import OrderedDict

font = cv2.FONT_HERSHEY_SIMPLEX

oldNew = int(sys.argv[4]) #  0:old 1:new

if oldNew == 1:
    gMapDict = {"1":"21","2":"24","3":"23","4":"25","5":"26","6":"22"}
    indexMap = {"21":"mot","24":"Belt","23":"Accessories","25":"Other", "26":"TissueBox", "22":"SunVisor"}
    ctypeMap = OrderedDict()
    ctypeMap['21'] = 'mot'
    ctypeMap['24'] = 'Belt' 
    ctypeMap['23'] = 'Accessiries' 
    ctypeMap['25'] = 'Other'       
    ctypeMap['26'] = 'TissueBox' 
    ctypeMap['22'] = 'SunVisor'  
else:
    gMapDict = {"1":"1","2":"2","3":"3","4":"4","5":"5","6":"6"}
    indexMap = {"1":"mot","2":"Belt","3":"Accessories","4":"Other", "5":"TissueBox", "6":"SunVisor"}
    ctypeMap = OrderedDict()
    ctypeMap['1'] = 'mot'
    ctypeMap['2'] = 'Belt'
    ctypeMap['3'] = 'Accessiries'
    ctypeMap['4'] = 'Other'
    ctypeMap['5'] = 'TissueBox'
    ctypeMap['6'] = 'SunVisor'

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

# main
def draw(gt, rt, fobj, cType, virtual):

    errorNums = 0
    lostNums = 0
    totalNums = 0

    for img in gt:

        if virtual == 1:
            im = cv2.imread("images/" + img)
            if im is None:
                print('img:', img)
            im_copy = im.copy()

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
            for gRoi in gRois:
                gx, gy, gw, gh, gts = gRoi
                if gts == cType:
                    lostNums += 1
            continue

        #print('rRois:', rRois)
        #print('gRois:', gRois)
        drawRois = []

        for rRoi in rRois:
            rx, ry, rw, rh, rts = rRoi

            flag = 0
            index = 0
            ioBest = 0
            ioIndex = 0
            for gRoi in gRois:
                gx, gy, gw, gh, gts = gRoi
                io = iou([int(rx),int(ry),int(rw),int(rh)],[int(gx),int(gy),int(gw),int(gh)])
                if io > 0 and rts == gts and gts == cType:
                    flag = 1
                    #del gRois[index]
                    #break
                    if io > ioBest:
                        ioBest = io
                        ioIndex = index
                index += 1

            if flag == 0:
                drawRois.append([rx,ry,rw,rh,rts])
            else:
                del gRois[ioIndex]
                
        # error detect 
        errorNum = 0
        for oRoi in drawRois:
            x, y, w, h, t = oRoi
            if t != cType:
                continue
            errorNum += 1

            if virtual == 1:
                cv2.rectangle(im_copy,(int(x),int(y)),(int(x) + int(w), int(y) + int(h)),(0,0,255),1)
                cv2.putText(im_copy, t, (int(x)+int(w), int(y)+int(h)/2), font, 0.5, (0, 0, 255), 1)
        errorNums += errorNum
 
        # lost detect
        lostNum = 0
        for gRoi in gRois:
            x, y, w, h, t = gRoi
            if t != cType:
                continue
            lostNum += 1

            if virtual == 1:
                cv2.rectangle(im_copy,(int(x),int(y)),(int(x) + int(w), int(y) + int(h)),(0,255,0),1)
                cv2.putText(im_copy, t, (int(x)+int(w), int(y)+int(h)/2), font, 0.5, (0, 255, 0), 1)
        lostNums += lostNum

        
        if errorNum != 0 or lostNum != 0:
            if virtual == 1:
                if not os.path.exists(cType):
                    os.system("mkdir {}".format(cType))
                cv2.imwrite(cType + "/" + img, im_copy)

        #print('totalNums:', totalNums)
        #print('lostNums:', lostNums)
        #print('errNums:', errorNums)
        #raw_input()

    print('totalNums:', totalNums)
    print('lostRatioo(lostNums/totalNums):', 1.0*lostNums/totalNums)
    print('errRatio(errorNums/totalNums):', 1.0*errorNums/totalNums)
    errRatio = 1.0*errorNums/totalNums
    lostRatio = 1.0*lostNums/totalNums
    f1 = 2*errRatio*lostRatio/(lostRatio+errRatio)
    #fobj.write("{},{},{},{}{}".format(scene, round(lostRatio, 4), round(errRatio, 4), round(f1, 4), os.linesep))
    fobj.write("{}({}),{},{},{}{}".format(ctypeMap[ctype], totalNums, round(lostRatio, 4), round(errRatio, 4), round(f1, 4), os.linesep))
    
if __name__ == "__main__":
    fobj = open('marker.csv', 'w')
    gt = pgtDict(sys.argv[1])
    rt = prtDict(sys.argv[2])
    ctypes = ctypeMap.keys()
    virtual = int(sys.argv[3])
    for ctype in ctypes:
        print('ctype:', indexMap[ctype])
        os.system("rm -rf {}".format(ctype))
        draw(gt, rt, fobj, ctype, virtual)
    fobj.close()
