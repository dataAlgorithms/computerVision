import math
from imutils import build_montages
import cv2
import json
import sys
import os
import re
import numpy as np

font = cv2.FONT_HERSHEY_SIMPLEX

def iou(b1, b2):
    b1 = [int(item) for item in b1]
    b2 = [int(item) for item in b2]

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

def getMost(arr):

    import numpy as np
    arr=np.array(arr)
    tu=sorted([(np.sum(arr==i),i) for i in set(arr.flat)])
    print(tu)
    #print('the most item is {1} has {0} num'.format(*tu[-1]))
    return tu

gt = sys.argv[1]
pt = sys.argv[2]

# build gt dict
gtFaceDict = {}
gtFrameDict = {}

with open(gt) as g:
    data = json.load(g)

i = 0
uids = []
imgs = []
for key in sorted(data, key=lambda item:int(item)):

    if int(key) > 1200:
        break

    for item in data[key]:
        uid = data[key][item]['pid']
        iclass = data[key][item]['class']
        pts = data[key][item]['pt']
        pts = [str(item) for item in pts]

        if iclass != 5:
            continue

        frame = "%06d" % int(key)
        
        if frame not in gtFrameDict:
            gtFrameDict[frame] = [str(uid) + ':' + '_'.join(pts)]
        else:
            gtFrameDict[frame].append(str(uid) + ':' + '_'.join(pts))

        if uid not in gtFaceDict:
            gtFaceDict[uid] = [frame + ':' + '_'.join(pts)]
        else:
            gtFaceDict[uid].append(frame + ':' + '_'.join(pts))

f = open('gtFrame', 'w')
for frame in gtFrameDict:
    f.write("{} {}{}".format(frame, gtFrameDict[frame], os.linesep))
f.close()

# build pt dict
ptFaceDict = {}

with open(pt) as p:
    lines = (line.strip() for line in p)
    for line in lines:
        lineItems = line.split(" ")
        img = os.path.basename(lineItems[0]).split(".")[0]
        others = lineItems[1:]
        pairNum = 8
        loopNum = len(others)/pairNum
        for i in range(loopNum):
            startIndex = i*pairNum
            endIndex = (i+1)*pairNum
            item = others[startIndex:endIndex]
            uid,c,x,y,w,h,r,q = item
            if uid not in ptFaceDict:
                ptFaceDict[uid] = [img + ':' + '_'.join([x,y,w,h]) + ':' + r]
            else:
                ptFaceDict[uid].append(img + ':' + '_'.join([x,y,w,h]) + ':' + r)

            #print(ptFaceDict)
            #raw_input()
# count
realFaceNum  = len(gtFaceDict)
   
trackNum = sum([len(ptFaceDict[key]) for key in ptFaceDict])
errNum = 0
print('ptFaceDictLen:', len(ptFaceDict))

hitGt = []
errList = []
delKeys = []
lostList = []
connRatios = []
for index, pid in enumerate(ptFaceDict):
    #print('index:', index)

    ridPC = []
    exitFlag = 0
    faceNum = 0
    for pItems in ptFaceDict[pid]:

        print(pItems)

        flag = 0

        #if len(gtFaceDict) == 0:
        #    break
   
        frameP, ptsP, ridP = pItems.split(":")
        ptsP = ptsP.split("_")
        gItems = gtFrameDict[frameP]

        for inIndex, item in enumerate(gItems):
            uidG, ptsG = item.split(":")
            ptsG = ptsG.split("_")
            ious = iou(ptsP, ptsG)
            if ious >= 0.5:
                flag = 1
                faceNum += 1
                hitGt.append(uidG)
                del gItems[inIndex]
                if uidG not in delKeys:
                    delKeys.append(uidG)
                if ridP != '0':
                    ridPC.append(ridP)
                break

        if exitFlag == 1:
            break

    if flag == 1:
        faceNum = faceNum
        ridNum = len(ridPC)
        try:
            ridMost = getMost(ridPC)[-1][0]
            print(ridMost)
            connRatio = 1.0*ridMost/faceNum
            connRatios.append(connRatio)
        except:
            pass
        #print('faceNum:{} ridMost:{}'.format(faceNum, ridMost))
        #raw_input()

print('avg connect ratio:', np.mean(connRatios))
