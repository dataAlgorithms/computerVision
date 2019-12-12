import cv2
import re
import os
import sys
import numpy as np

gt = sys.argv[1]
pt = sys.argv[2]
vf = int(sys.argv[3])

def draw_circle(img, point, point_color):
    point_size = 1
    thickness = 4  # 可以为 0 、4、8
    cv2.circle(img, point, point_size, point_color, thickness)
    cmd = os.path.dirname(os.path.abspath(__file__)) + '/'
    # cv2.imwrite(cmd + save_name, img)

def drawLmk(gDict, pDict):
    for img in gDict:
        im = cv2.imread("/mnt/cephfs/testData/fms/personPose/" + img)
        gPoint = list(gDict[img])
        for i in range(0, len(gPoint), 2):
            point = tuple([int(eval(dot)) for dot in gPoint[i:i+2]])
            draw_circle(im, point, (0,255,0))

        pPoint = list(pDict[img])
        for i in range(0, len(pPoint), 2):
            point = tuple([int(eval(dot)) for dot in pPoint[i:i+2]])
            draw_circle(im, point, (255,0,0))

        cv2.imwrite("poseAll/{}".format(img), im)

def nrmse(array_vega, array_gt):  # 标准均方根误差
    #mse = np.mean((array_vega - array_gt) ** 2)
    mse = np.linalg.norm(array_gt - array_vega, axis=1).mean()
    mse = round(mse, 2)
    return mse

def get_array(list_point):
    # list_point = list(map(float, list_point))
    list_point = list(map(float, list_point))
    list_point = list(map(int, list_point))
    array_point = np.array(list_point)

    return array_point

gDict = {}
with open(gt) as g:
    lines = (line.strip() for line in g)
    for line in lines:
        img,nose1,nose2,left_eye1,left_eye2,right_eye1,right_eye2,left_ear1,left_ear2,right_ear1,right_ear2,left_shoulder1,left_shoulder2,right_shoulder1,right_shoulder2,left_elbow1,left_elbow2,right_elbow1,right_elbow2,left_wrist1,left_wrist2,right_wrist1,right_wrist2,left_hip1,left_hip2,right_hip1,right_hip2,left_knee1,left_knee2,right_knee1,right_knee2,left_ankle1,left_ankle2,right_ankle1,right_ankle2,_,_,_,_,Leftheel1,Leftheel2,Rightheel1,Rightheel2,_,_,_,_,_,_  = line.split(",")
        img = os.path.basename(img)
        gDict[img] = nose1,nose2,left_eye1,left_eye2,right_eye1,right_eye2,left_ear1,left_ear2,right_ear1,right_ear2,left_shoulder1,left_shoulder2,right_shoulder1,right_shoulder2,left_elbow1,left_elbow2,right_elbow1,right_elbow2,left_wrist1,left_wrist2,right_wrist1,right_wrist2,left_hip1,left_hip2,right_hip1,right_hip2,left_knee1,left_knee2,right_knee1,right_knee2,left_ankle1,left_ankle2,right_ankle1,right_ankle2,Leftheel1,Leftheel2,Rightheel1,Rightheel2

pDict = {}
with open(pt) as p:
    lines = (line.strip() for line in p)
    for  line in lines:
        img,nose1,nose2,_,left_eye1,left_eye2,_,right_eye1,right_eye2,_,left_ear1,left_ear2,_,right_ear1,right_ear2,_,left_shoulder1,left_shoulder2,_,right_shoulder1,right_shoulder2,_,left_elbow1,left_elbo2,_,right_elbow1,right_elbow2,_,left_wrist1,left_wrist2,_,right_wrist1,right_wrist2,_,left_hip1,left_hip2,_,right_hip1,right_hip2,_,left_knee1,left_knee2,_,right_knee1,right_knee2,_,left_ankle1,left_ankle2,_,right_ankle1,right_ankle2,_,Leftbigtoe1,Leftbigtoe2,_,Leftsmalltoe1,Leftsmalltoe2,_,Leftheel1,Leftheel2,_,Rightbigtoe1,Rightbigtoe2,_,Rightsmalltoe1,Rightsmalltoe2,_,Rightheel1,Rightheel2,_  = line.split(" ")
        img = os.path.basename(img)
        pDict[img] = nose1,nose2,left_eye1,left_eye2,right_eye1,right_eye2,left_ear1,left_ear2,right_ear1,right_ear2,left_shoulder1,left_shoulder2,right_shoulder1,right_shoulder2,left_elbow1,left_elbow2,right_elbow1,right_elbow2,left_wrist1,left_wrist2,right_wrist1,right_wrist2,left_hip1,left_hip2,right_hip1,right_hip2,left_knee1,left_knee2,right_knee1,right_knee2,left_ankle1,left_ankle2,right_ankle1,right_ankle2,Leftheel1,Leftheel2,Rightheel1,Rightheel2

mses = []
for img in gDict:
    gt = get_array(gDict[img])
    pt = get_array(pDict[img])
    mse = nrmse(gt.reshape(19, 2),  pt.reshape(19, 2))
    mses.append(mse)

print('avgMse:', np.array(mses).mean())
if vf == 1:
   drawLmk(gDict, pDict) 
