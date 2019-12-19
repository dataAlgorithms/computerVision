#!/usr/bin/env python
#!coding=utf-8

import sys
import os
import base64
import numpy as np
import sys
import math

def array2feature(array):
    aList = [eval(i) for i in array.split(" ")]
    featureFloat = np.array(aList,dtype=np.float32)
    array = np.array(featureFloat)
    #array = array/math.sqrt(2)
    feature = base64.b64encode(array)

    return feature


def feaFormat(feaFile):

    nObj = open(feaFile + ".new", "w+")

    with open(feaFile) as ff:
        for line in ff:
            line = line.split(".jpg ")
            jpg = line[0] + ".jpg"
            arrFea = line[1].strip("\r\n ")
            fea = array2feature(arrFea)
            nObj.write("{} {}{}".format(jpg, fea, os.linesep))
    nObj.close()

if __name__ == '__main__':
    feaFile = sys.argv[1]
    feaFormat(feaFile) 
