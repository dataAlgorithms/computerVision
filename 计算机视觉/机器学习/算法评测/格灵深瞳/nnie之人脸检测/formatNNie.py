#!/usr/bin/env python

import os
import sys

aDict = {}
fname = sys.argv[1]
nObj = open("new." + fname, "w")
icount = 0
ncount = 0
aList = []
with open(fname) as fn:
    for line in fn:
        icount += 1
        line = line.strip("\n").split(".jpg ")
        if len(line) == 1:
            img = line[0]
            aDict[img] = []
            continue
        img, others = line
        others = others.split(" ")
        per5 = len(others)/6
        if img + ".jpg" not in aDict:
            aDict[img + ".jpg"] = []
        for i in range(per5):
            c, t, x, y, w, h = others[i*6:6*(i+1)]
            if float(c) < 0.4:
                continue
            x = int(float(x))
            y = int(float(y))
            w = int(float(w))
            h = int(float(h))
            score = c
            aDict[img + ".jpg"].append([x,y,w,h,score])
        ncount += 1

for img in aDict:
    if len(img) == 0:
        print(img)
        nObj.write(img)
    else:
        nObj.write("{} ".format(img))
        for item in aDict[img]:
            x, y, w, h, score = item
            nObj.write("{} {} {} {} {} ".format(x, y, w, h, score))
    nObj.write(os.linesep)

nObj.close()

os.system("sort {} -o {}".format("new."+fname, "new."+fname))
