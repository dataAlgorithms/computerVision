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

gt = sys.argv[1]
pt = sys.argv[2]
pcnType = int(sys.argv[3]) # 1:vehicle, 2:people, 3:bicycle, 4:tricycle

pcnDict = {1:'car_attr', 2:'ped_attr', 3:'transportation_vehicle2', 4:'transportation_vehicle3'}

# build gt dict
gtFaceDict = {}
gtFrameDict = {}

i = 0
uids = []
imgs = []

with open(gt) as g:
    data = json.load(g)
for key in sorted(data, key=lambda item:int(item)):

    if data[key] is None:
        continue

    for item in data[key]:
        uid = data[key][item]['id']
        iclass = data[key][item]['class']
        pts = data[key][item]['pt']
        pts = [str(item) for item in pts]

        if iclass != pcnType:
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

'''
with open(gt) as g:
    for line in g:
        line = line.strip("\r\n")
        img, uid, iclass, x, y, w, h = line.split(",")
        pts = [int(float(x)), int(float(y)), int(float(w)), int(float(h))]
        pts = [str(item) for item in pts]
        if int(iclass) != 1:
            continue
       
        key = img
 
        frame = "%06d" % int(key)
        
        if frame not in gtFrameDict:
            gtFrameDict[frame] = [str(uid) + ':' + '_'.join(pts)]
        else:
            gtFrameDict[frame].append(str(uid) + ':' + '_'.join(pts))

        if uid not in gtFaceDict:
            gtFaceDict[uid] = [frame + ':' + '_'.join(pts)]
        else:
            gtFaceDict[uid].append(frame + ':' + '_'.join(pts))
'''
f = open('gtFrame', 'w')
for frame in gtFrameDict:
    f.write("{} {}{}".format(frame, gtFrameDict[frame], os.linesep))
f.close()

# build pt dict
ptFaceDict = {}

with open(pt) as p:
    for line in p:
        if pcnDict[pcnType] not in line:
            continue
        match = re.search(r"(\d+)\.jpg id\((\d+)\) rect\((\d+) (\d+) (\d+) (\d+)\)", line, re.DOTALL)
        img, uid, x, y, w, h = match.group(1), match.group(2), match.group(3), match.group(4), match.group(5), match.group(6)
        if uid not in ptFaceDict:
            ptFaceDict[uid] = [img + ':' + '_'.join([x,y,w,h])]
        else:
            ptFaceDict[uid].append(img + ':' + '_'.join([x,y,w,h]))

# count
realFaceNum  = len(gtFaceDict)
#for key in ptFaceDict:
#    print('key:{} value:{}'.format(key, ptFaceDict[key]))
   
trackNum = sum([len(ptFaceDict[key]) for key in ptFaceDict])
errNum = 0

hitGt = []
errList = []
delKeys = []
lostList = []
for index, pid in enumerate(ptFaceDict):
    #print('index:', index)
    pItems = ptFaceDict[pid][0]
    flag = 0
    #print('pItems:', len(pItems))

    if len(gtFaceDict) == 0:
         break
    
    frameP, ptsP = pItems.split(":")
    ptsP = ptsP.split("_")

    try:
        gItems = gtFrameDict[frameP]
    except:
        errNum += 1
        errList.append([pid, ptFaceDict[pid]])
        continue

    for inIndex, item in enumerate(gItems):
        uidG, ptsG = item.split(":")
        ptsG = ptsG.split("_")
        ptsG = [int(float(item)) for item in ptsG]
        ious = iou(ptsP, ptsG)
        if ious >= 0.5:
            flag = 1
            hitGt.append(uidG)
            del gItems[inIndex]
            if uidG not in delKeys:
                delKeys.append(uidG)
            break

    if flag == 0:
        errNum += 1
        errList.append([pid, ptFaceDict[pid]])

for uid in gtFaceDict:
    if str(uid) not in delKeys:
        lostList.append([uid, gtFaceDict[uid]])

'''
for item in lostList:
    print(item[0], item[1])
    raw_input()
'''
'''
print('index:', index)
print('hitGtLen:', len(hitGt))
print('hitGtLen:', len(set(hitGt)))
print('errList:', errList)
print('len delKeys:', len(delKeys))
print('lostList:', len(lostList))
'''

e = open('err.txt', 'w')
for item in errList:
    e.write("{}{}".format(item, os.linesep))
e.close()

l = open('lost.txt', 'w')
for item in lostList:
    l.write("{}{}".format(item, os.linesep))
l.close()

realFaceNum = realFaceNum
trackNum = trackNum
errDetectNum = len(errList)
repeatNum = len(hitGt)-len(set(hitGt))
finalTrackNum = trackNum-errDetectNum-repeatNum
lossNum = len(lostList)

capRatio=1.0*finalTrackNum/realFaceNum
repeatRatio=(trackNum-errDetectNum)*1.0/finalTrackNum
errRatio=1.0*errDetectNum/trackNum
lostRatio=1.0*lossNum/realFaceNum

print('realNum:', realFaceNum)
print('trackNum:', trackNum)
print('errDetectNum:', errDetectNum)
print('repeatNum:', repeatNum)
print('finalTrackNum:', finalTrackNum)
print('lossNum:', lossNum)
print('capRatio:', capRatio)
print('repeatRatio:', repeatRatio)
print('errRatio:', errRatio)
print('lostRatio:', lostRatio)

print('finalTrackNum:', len(set(hitGt)))
print('repeatNum:', len(hitGt)-len(set(hitGt)))
print('lostNum:', len(lostList))
print('errNum:', len(errList))
print('realFaceNum:', realFaceNum)
print('trackNum:', trackNum)

'''
# draw err
for item in errList:
    pid, stream_roi = item
    stream, roi = stream_roi[0].split(":")
    x, y, w, h = roi.split("_")
    im = cv2.imread("/data/zhouping/dgnnie/3516/dgnnie_3516_v3.5.1.a4.safenet/Release/images/3_1920x1080/{}.jpg".format(stream))
    im_copy = im.copy()
    gap = 0
    y1 = int(y)-gap
    if y1 < 0:
        y1 = 0

    y2 = int(y)+int(h)+gap
    x1 = int(x)-gap
    if x1 < 0:
        x1 = 0

    x2 = int(x)+int(w)+gap

    cropped = im_copy[y1:y2, x1:x2] # y1,y2,x1,x2
    cv2.imwrite("errDraw/{}_{}_{}_{}_{}_{}.jpg".format(stream, x,y,w,h,pid), cropped)


# draw lost
for item in lostList:
    pid, stream_rois = item
    images = []
    for stream_roi in stream_rois:
        stream, roi = stream_roi.split(":")
        x, y, w, h = roi.split("_")
        im = cv2.imread("/data/zhouping/dgnnie/3516/dgnnie_3516_v3.5.1.a4.safenet/Release/images/3_1920x1080/{}.jpg".format(stream))
        im_copy = im.copy()

        gap = 0
        y1 = int(y)-gap
        if y1 < 0:
            y1 = 0

        y2 = int(y)+int(h)+gap
        x1 = int(x)-gap
        if x1 < 0:
            x1 = 0

        x2 = int(x)+int(w)+gap

        cropped = im_copy[y1:y2, x1:x2] # y1,y2,x1,x2
        cv2.putText(cropped, stream, (int(10), int(30)), font, 1, (0, 0, 255), 2)
        images.append(cropped)

    if len(images) < 10:
        w = len(images)
        h = 1
    else:
        w = 10
        h = 1.0*len(images)/w
        h = int(math.ceil(h))
        if h == 0:
            h = 1

    montage = build_montages(images, (200, 200), (w, h))[0]
    cv2.imwrite("lostDraw/{}.jpg".format(pid), montage)
'''
