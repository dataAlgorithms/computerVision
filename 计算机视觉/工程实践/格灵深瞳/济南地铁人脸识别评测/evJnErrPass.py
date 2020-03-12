#!/usr/bin/env python
#!coding=utf-8

import fnmatch
import re
import os
import string
import requests
import json
import psycopg2
import numpy as np
import base64
from optparse import OptionParser
import math
import sys
import random
from functools import partial
from contextlib import contextmanager
import multiprocessing

@contextmanager
def poolcontext(*args, **kwargs):
    pool = multiprocessing.Pool(*args, **kwargs)
    yield pool
    pool.terminate()

def gen_find(filepat="*", top=None):
    '''
    Find all filenames in a directory tree that match a shell wildcard pattern
    '''
    for path, dirlist, filelist in os.walk(top):
        for name in fnmatch.filter(filelist, filepat):
            yield os.path.join(path,name)

def postDeepNet(img, threshold):

    threshold = threshold/100.0
    tmpres = os.popen("""curl -s --location --request POST 'http://192.168.2.227:3154/api/data/flow/compare' --header 'Content-Type: multipart/form-data' --header 'access_key: 7966023d-0eee-4104-9c2f-8efa4279442a' --header 'secret_key: 9e7b1900-302b-4059-bcaa-047262f3ca3b' --header 'authkey: dp-auth-v0' --form 'CaptureImage=@{}' --form 'DeviceID=JiNan-in-01' --form 'RepoID=4b02f98f-d43a-4a77-b188-4984a0a8e7a3' --form 'TimeStamp=1583993686300' --form 'Confidence={}'""".format(img, threshold)).readlines()
    response = json.loads(tmpres[0])
    code = response["Code"]
    data = response['Data']
    if int(code) != 1:
        return None
    else:
        try:
            tId = data['Faces'][0]['Tags'][0]['ExternalID']
            tScore = data['Faces'][0]['Tags'][0]['Confidence']
            return tId.lstrip("suanfa-"), tScore
        except:
            return None

def post_request(url, source):
    jsource = json.dumps(source)
    resp = requests.post(url, data = jsource)
    if resp.content == "":
        return None,resp.status_code
    else:
        rdict = json.loads(resp.content)
        return rdict,resp.status_code

def androidTopFast(threshold, line):

    fObj = open('err.txt', 'a')

    wrongFZ = 0
    rightFZ = 0
    tp = fp = tn = fn = 0
    feature_id = line
    idScore = postDeepNet(feature_id, threshold)
    #print('feature_id:', feature_id)
    #print('idScore:',  idScore)
    if 'sence923' in feature_id:

        if idScore is None:
            fn += 1
        else:
            top1Candidate = idScore[0]
            top1Score = idScore[1]*100
            #print(feature_id)
            headC = os.path.basename(top1Candidate).split(".")[0]
            headQ = os.path.basename(feature_id).split(".")[0]
            if headC != headQ:
                fn += 1
                wrongFZ += 1
                fObj.write('222 feature_id:{} top1Candidate:{} top1Score:{}{}'.format(feature_id, top1Candidate, top1Score, os.linesep))
            elif headC == headQ and top1Score < threshold:
                fn += 1
                wrongFZ += 1
                fObj.write('333 feature_id:{} top1Candidate:{} top1Score:{}{}'.format(feature_id, top1Candidate, top1Score, os.linesep))
            elif headC == headQ and top1Score >= threshold:
                tp += 1
                rightFZ += 1
            else:
                print('else 1111111111111')
    if 'black10w' in feature_id:

        if idScore is None:
            tn += 1
        else:
            top1Candidate = idScore[0]
            top1Score = idScore[1]*100

            if top1Score >= threshold:
                #print('feature_id:', feature_id)
                #print('feature_str:', feature_str)
                #print('wrongFZ:', wrongFZ)
                #raw_input()
                #if 'licenseCard100' in top1Candidate and top1Score >= threshold:
                fObj.write('444 feature_id:{} top1Candidate:{} top1Score:{}{}'.format(feature_id, top1Candidate, top1Score, os.linesep))
                wrongFZ += 1
                fp += 1
            else:
                tn += 1
    fObj.close()

    return wrongFZ, rightFZ, tp, fp, tn, fn

if __name__ == "__main__":
    #androidTop(repoIds, featureFile)

    #idScore = postDeepNet("0000002232.jpeg", 0.8)
    #print('idScore:', idScore)
    #sys.exit(0)

    # get the 10w distrib
    allDis = gen_find("*.jpg", "/data/dgtestdata/algoSdk/face/faceFeature/bankVip/black10w")
    # get the query 
    allQuerys = gen_find("*.jpeg", "/data/dgtestdata/algoSdk/face/faceFeature/cardSence/sence923")

    feaLines = list(allQuerys)+list(allDis)
    feaLines = feaLines
    print('feaLines:', len(feaLines))

    nObj = open('pr.txt', 'w')

    thresholds = range(85, 101)
    for threshold in thresholds:
        with poolcontext(processes=48) as pool:
            results = pool.map(partial(androidTopFast, threshold), feaLines)

        wrongFZ = 0
        wrongFM = len(feaLines)
        rightFZ = 0
        rightFM = 4373
        tps = fps = tns = fns = 0
        for result in results:
            wrongFZ += result[0]
            rightFZ += result[1]
            tps += result[2]
            fps += result[3]
            tns += result[4]
            fns += result[5]

        fpr = 1.0 * fps / (fps+tns)
        tpr = 1.0 * tps / (tps+fns)

        wrongRecRate = 1.0*wrongFZ/wrongFM
        precision = 1.0*rightFZ/rightFM
        #print("threshold:{} fpr:{} tpr:{}".format(threshold, fpr, tpr))
        nObj.write('threshold:{} wrongRecRate:{} precision:{}{}'.format(threshold, fpr, tpr, os.linesep))
    nObj.close()
