#!/usr/bin/env python
#!coding=utf-8

import numpy as np
import sys
import os
import copy
import cv2
import re
from PIL import Image, ImageDraw, ImageFont

font = cv2.FONT_HERSHEY_SIMPLEX
fontpath = "./NotoSansCJK-Black.ttc" #https://pan.baidu.com/s/1y8i8fEg5Hk2UQo6G6dz_BA
pfont = ImageFont.truetype(fontpath, 20)

# color dict
carColorDict = {'2':'car_color_blue', '5': 'car_color_gray', '3':'car_color_brown',
             '8':'car_color_purple', '12': 'car_color_yellow', '7':'car_color_pink',
             '1':'car_color_black', '6':'car_color_orange', '4':'car_color_green',
             '11':'car_color_white', '10':'car_color_silver', '9':'car_color_red'}

# carbranc dict
carBrandDict = {}
with open("CarBrand_cls_2_label") as g:
    for line in g:
        line = line.strip("\r\n")
        stype, sid = line.split(" ")
        carBrandDict[sid] = stype

# plate color dict
plateColorDict = {
    "-1": "未知",
    "1": "蓝色",
    "2": "黄色",
    "3": "白色",
    "4": "黑色",
    "5": "绿色",
    "6": "黄绿色",
    "7": "渐变绿"
}

# face attr
glassDict = {"1":"no_glass", "2":"glass", "3":"sunglass"}
hatDict = {"1":"no_hat", "2":"hat", "3":"scraft"}
maskDict = {"1":"no_mask", "2":"mask"}
genderDict = {"1":"man", "2":"woman", "3":"unknown"}
ethnicDict = {"1":"non_uygur", "2":"uygur", "3":"non_uygur", "4":"non_uygur", "5":"non_uygur", "6":"non_uygur"}

# build result mapping dict
def prtDict(rtName, ftName):

    rtDict = {}
    with open(rtName) as rt:
        lines = (line.strip() for line in rt)
        for line in lines:
            img = line.split(" ")[0]

            if img not in rtDict:
                rtDict[img] = {'ped':[],'car':[], 'face':[]}

            idMatch = re.search(r"id\((\d+)\)", line)
            rectMatch = re.search(r"rect\((\d+) (\d+) (\d+) (\d+)\)", line)
            qMatch = re.search(r"quality\(([\d\.]+)\)", line) 

            tid = idMatch.group(1)
            x, y, w, h = rectMatch.group(1), rectMatch.group(2),rectMatch.group(3),rectMatch.group(4)
            q = qMatch.group(1)

            if 'ped_attr' in line:
                rtDict[img]['ped'].append([x,y,w,h,q,tid])
            elif 'car_attr' in line:
                carbrandMatch = re.search(r"attr_id:-1\(year\) attr_matrix_value: (\d+) conf: ([\.0-9e-]+)",  line)
                carcolorMatch = re.search(r"car color\) attr_matrix_value: (\d+) conf: ([\d\.]+)", line)
                plateColorMatch = re.search(r"attr_id:21\(plate color\) attr_matrix_value: (\d+) conf:", line, re.IGNORECASE)
                plateCharMatch = re.search(r"\[attr_id:-1\(plate literal\) attr_value: (.*?) ", line, re.IGNORECASE)

                if carbrandMatch:
                    carbrand = carBrandDict[carbrandMatch.group(1)]
                else:
                    carbrand = -1

                if carcolorMatch:
                    carcolor = carColorDict[carcolorMatch.group(1)]
                else:
                    carcolor = -1

                if plateColorMatch:
                    platecolor = plateColorDict[plateColorMatch.group(1)]
                else:
                    platecolor = -1

                if plateCharMatch:
                    plateChar = plateCharMatch.group(1)
                else:
                    plateChar = -1

                rtDict[img]['car'] = [[x, y, w, h, carbrand, carcolor, platecolor, plateChar]]

    with open(ftName) as rt:
        lines = (line.strip() for line in rt)
        for line in lines:
            img = line.split(" ")[0]

            if img not in rtDict:
                rtDict[img] = {'ped':[],'car':[], 'face':[]}

            idMatch = re.search(r"id\((\d+)\)", line)
            rectMatch = re.search(r"rect\((\d+) (\d+) (\d+) (\d+)\)", line)
            qMatch = re.search(r"quality\(([\d\.]+)\)", line)

            tid = idMatch.group(1)
            x, y, w, h = rectMatch.group(1), rectMatch.group(2),rectMatch.group(3),rectMatch.group(4)
            q = qMatch.group(1)

            match = re.search(r"(.*.jpe?g).*?\[attr_id:1\(age\) attr_(?:matrix_)?value: (\d+) conf: 1\] \[attr_id:16\(gender\) attr_(?:matrix_)?value: (\d+) conf: [\d\.]+\] \[attr_id:6\(mask\) attr_(?:matrix_)?value: (\d+) conf: [\d\.]+\] \[attr_id:4\(hat\) attr_(?:matrix_)?value: (\d+) conf: [\d\.]+\] \[attr_id:3\(glass\) attr_(?:matrix_)?value: (\d+) conf: [\d\.]+\] \[attr_id:19\(ethnic\) attr_(?:matrix_)?value: (\d+) conf: [\d\.]+\]", line)
            if match:
                age = match.group(2)
                gender = genderDict[match.group(3)]
                mask = maskDict[match.group(4)]
                hat = hatDict[match.group(5)]
                glass = glassDict[match.group(6)]
                ethnic = ethnicDict[match.group(7)]
            else:
                age = -1
                gender = -1
                mask = -1
                hat = -1
                glass = -1
                ethnic = 01    
                            
            rtDict[img]['face'] = [[x, y, w, h, age,gender, mask, hat, glass, ethnic]]    

    return rtDict

# main
def draw(rt):

    for img in rt:

        im = cv2.imread(re.sub("/nfs/Release", "..", img))
        im_copy = im.copy()

        ped = rt[img]['ped']
        if len(ped) == 0:
            pass
        else:
            rRois = ped
            print(rRois)    
            for rRoi in rRois:
                x, y, w, h, q, t = rRoi
                cv2.rectangle(im_copy,(int(x),int(y)),(int(x) + int(w), int(y) + int(h)),(0,0,255),2)
                #cv2.putText(im_copy, 'id:'+t, (int(x), int(y)), font, 1, (0, 0, 255), 2)
                #cv2.putText(im_copy, str(round(float(q),2)), (int(x), int(y)), font, 0.5, (0, 0, 255), 1)

        car = rt[img]['car']
        if len(car) == 0:
            pass
        else:
            rRois = car
            print(rRois)
            for rRoi in rRois:
                x, y, w, h, carbrand, carcolor, platecolor, platechar = rRoi
                cv2.rectangle(im_copy,(int(x),int(y)),(int(x) + int(w), int(y) + int(h)),(0,0,255),2)               
                
                if carbrand != -1:
                    img_pil = Image.fromarray(im_copy)
                    draw = ImageDraw.Draw(img_pil)
                    b, g, r, a = 255, 0, 0, 0
                    draw.text((int(x), int(y)), carbrand.strip("\r\n").decode("utf-8"), font = pfont, fill = (b, g, r, a))
                    im_copy = np.array(img_pil)
                else:
                    cv2.putText(im_copy, 'no carbrand', (int(x), int(y)), font, 0.5, (0, 0, 255), 1)  

                if carcolor != -1:                                                                                                                                      
                    img_pil = Image.fromarray(im_copy)                                                                                                                  
                    draw = ImageDraw.Draw(img_pil)                                                                                                                      
                    b, g, r, a = 255, 0, 0, 0                                                                                                                           
                    draw.text((int(x), int(y)+40), carcolor.strip("\r\n").decode("utf-8"), font = pfont, fill = (b, g, r, a))                                              
                    im_copy = np.array(img_pil)                                                                                                                         
                else:                                                                                                                                                   
                    cv2.putText(im_copy, 'no carcolor', (int(x), int(y)+40), font, 0.5, (0, 0, 255), 1)  

                if platecolor != -1:                                                                                                                                      
                    img_pil = Image.fromarray(im_copy)                                                                                                                  
                    draw = ImageDraw.Draw(img_pil)                                                                                                                      
                    b, g, r, a = 255, 0, 0, 0                                                                                                                           
                    draw.text((int(x), int(y)+80), platecolor.strip("\r\n").decode("utf-8"), font = pfont, fill = (b, g, r, a))                                              
                    im_copy = np.array(img_pil)                                                                                                                         
                else:                                                                                                                                                   
                    cv2.putText(im_copy, 'no platecolor', (int(x), int(y)+80), font, 0.5, (0, 0, 255), 1)  


                if platechar != -1:                                                                                                                                      
                    img_pil = Image.fromarray(im_copy)                                                                                                                  
                    draw = ImageDraw.Draw(img_pil)                                                                                                                      
                    b, g, r, a = 255, 0, 0, 0                                                                                                                           
                    draw.text((int(x), int(y)+120), platechar.strip("\r\n").decode("utf-8"), font = pfont, fill = (b, g, r, a))                                              
                    im_copy = np.array(img_pil)                                                                                                                         
                else:                                                                                                                                                   
                    cv2.putText(im_copy, 'no platechar', (int(x), int(y)+120), font, 0.5, (0, 0, 255), 1)  

        face = rt[img]['face']
        if len(face) == 0:
            pass
        else:
            rRois = face                                                                                                                                                 
            print(rRois)                                                                                                                                                
            for rRoi in rRois:                                                                                                                                          
                x, y, w, h, age,gender, mask, hat, glass, ethnic = rRoi                                                                                            
                cv2.rectangle(im_copy,(int(x),int(y)),(int(x) + int(w), int(y) + int(h)),(0,0,255),2)                                                                   
                                                                                                                                                                        
                if age != -1:                                                                                                                                      
                    cv2.putText(im_copy, age, (int(x)+int(w), int(y)), font, 1, (255, 0, 0), 2)                                                                      
                    cv2.putText(im_copy, gender, (int(x)+int(w), int(y)+20), font, 1, (255, 0, 0), 2)                                                                      
                    cv2.putText(im_copy, mask, (int(x)+int(w), int(y)+40), font, 1, (255, 0, 0), 2)                                                                      
                    cv2.putText(im_copy, hat, (int(x)+int(w), int(y)+60), font, 1, (255, 0, 0), 2)                                                                      
                    cv2.putText(im_copy, glass, (int(x)+int(w), int(y)+80), font, 1, (255, 0, 0), 2)                                                                      
                    cv2.putText(im_copy, ethnic, (int(x)+int(w), int(y)+100), font, 1, (255, 0, 0), 2)                                                                      

        drawImg = os.path.basename(img)
        cv2.imwrite("draw/draw_" + drawImg, im_copy)
        cv2.destroyAllWindows()

if __name__ == "__main__":
    carName = sys.argv[1]
    faceName = sys.argv[2]
    rt = prtDict(carName, faceName)
    draw(rt)
