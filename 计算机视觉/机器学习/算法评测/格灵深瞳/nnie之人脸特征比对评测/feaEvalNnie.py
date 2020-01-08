#!/usr/bin/env python
#!coding=utf-8

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

featureFile = sys.argv[1]

@contextmanager
def poolcontext(*args, **kwargs):
    pool = multiprocessing.Pool(*args, **kwargs)
    yield pool
    pool.terminate()

def post_request(url, source):
    jsource = json.dumps(source)
    resp = requests.post(url, data = jsource)
    if resp.content == "":
        return None,resp.status_code
    else:
        rdict = json.loads(resp.content)
        return rdict,resp.status_code

# Repo class of Ranker2
class Ranker2Repo:
    def __init__(self, url):
        self.url = url

    def addRepo(self, source):
        resp_dict, _ret = post_request(self.url, source)
        #print '::Add repo result is as follow!'
        #print 'resp: ', json.dumps(resp_dict, indent=1)

    def queryRepo(self, source):
        resp_dict, _ret = post_request(self.url, source)
        #print '::Query repo result is as follow!'
        #print 'resp: ', json.dumps(resp_dict, indent=1)
        return resp_dict

    def deleteRepo(self, source):
        resp_dict, _ret = post_request(self.url, source)
        #print '::Delete repo result is as follow!'
        #print 'resp: ', json.dumps(resp_dict, indent=1)

    def updateRepo(self, source):
        resp_dict, _ret = post_request(self.url, source)
        return resp_dict
        #print '::update repo result is as follow!'
        #print 'resp: ', json.dumps(resp_dict, indent=1)

# Repo function
# Create repo session
def sessionRepo(rankerIp=None, rankerPort=None):

    # Assign the repo session
    repoSession = Ranker2Repo("http://%s:%s/rank/repo" % (rankerIp, rankerPort))

    return repoSession

# Add repo into ranker2
def addRepo(repoSession=None, repoId=None, capacity=None, level=3, featureLen=384, gPUThreads=[1,1,1,1]):

    # Assign ranker2 repo instance
    repo = repoSession

    # Do the repo add operation
    print '::Do the repo add'
    add_source = {"Context":{},"Repo":{"Operation":1,"RepoId":repoId,"Level":3,"FeatureLen":int(featureLen),"Capacity":int(capacity),"Params":{"GPUThreads":"%s" % gPUThreads}}}
    repo.addRepo(add_source)

# Feature class of Ranker2
class Ranker2Feature:
    def __init__(self, url):
        self.url = url

    def addFeature(self, source):
        resp_dict, _ret = post_request(self.url, source)
        print '::Add feature result is as follow!'
        #print 'resp: ', json.dumps(resp_dict, indent=1)

    def queryFeature(self, source):
        resp_dict, _ret = post_request(self.url, source)
        #print '::Query feature result is as follow!'
        #print 'resp: ', json.dumps(resp_dict, indent=1)

    def deleteFeature(self, source):
        resp_dict, _ret = post_request(self.url, source)
        #print '::Delete feature result is as follow!'
        #print 'resp: ', json.dumps(resp_dict, indent=1)

    def updateFeature(self, source):
        resp_dict, _ret = post_request(self.url, source)
        #print '::update feature result is as follow!'
        #print 'resp: ', json.dumps(resp_dict, indent=1)

# Feature function
# create feature based on length
def featureCreate(featureLen=None):

    # Check the feature lenght is valid or not
    if featureLen is None:
        print("Please input feature length, eg. 384")
        return

    # Create the the feature
    f_list = []
    ff_sum = 0.0
    for _ in range(featureLen):
        f = random.uniform(-1,1)
        ff = f*f
        f_list.append(f)
        ff_sum = ff_sum + ff
    t = math.sqrt(ff_sum)
    featureFloat = []
    for f in f_list:
        featureFloat.append(f/t)
    featureFloat = np.array(featureFloat,dtype=np.float32)
    featureString = base64.b64encode(featureFloat)
    print('featureString:', featureString)

    return featureString

# Add feature into repo
def featureAdd(featureSession=None, repoId=None, featureId=None, featureStr=None, time=None, location=None):

    add_source={"Features":{"Operation":1,"RepoId":str(repoId),"ObjectFeatures":[{"Feature":featureStr,"Time":int(time),"Id":str(featureId),"Location":str(location)}]},"Context":{"SessionId":"ss_743"}}
    featureSession.addFeature(add_source)

# Create ranker session
def sessionFeature(rankerIp=None, rankerPort=None):

    # Assign the feature session
    featureSession = Ranker2Feature("http://%s:%s/rank/feature" % (rankerIp, rankerPort))

    return featureSession

# Do the 1vN ranker
def vNRanker(rankUrl="http://192.168.2.19:6501/rank", repoId=None, location=None, featureStr=None):

     source = {"Params":{"RepoId":repoId,"Normalization":"false","Locations":str(location),"StartTime":"0","EndTime":"9999999999999"},"ObjectFeature":{"Feature":featureStr},"Context":{"SessionId":"test123"}}

     resp_dict,ret = post_request(rankUrl,source)
     return json.dumps(resp_dict, indent=1)
     try:
         return resp_dict["Candidates"][0]["Score"],resp_dict["Candidates"][0]["Id"]
     except:
         return None

def scoreTransform(score):

    #score = score * 1.34228188 - 0.14026846
    #score = score * 1.33333333 - 0.05533333
    #score = score * 1.72413887-0.33551787
    #score = score * 0.72228191+0.26776852
    score = score * 1.87192482-0.46675133
    if score > 1:
        score = 1
    if score < 0:
        score = 0

    return score

def array2feature(array):
    aList = [eval(i) for i in array.split()]
    featureFloat = np.array(aList,dtype=np.float32)
    array = np.array(featureFloat)
    array = array/math.sqrt(2)
    feature = base64.b64encode(array)
    #print 'feature:', feature

    return feature

def androidTopFast(threshold, line):

    fObj = open('err.txt', 'a')

    wrongFZ = 0
    rightFZ = 0
    tp = fp = tn = fn = 0
    #####feature_id = feature_line[0]
    #####feature_str = feature_line[1].strip("\r\n")

    #####vnResult = vNRanker(rankUrl="http://{}:{}/rank".format(rankIps, rankPorts), repoId=repoIds, location=0, featureStr=feature_str)
    #####ev_vnResults = eval(vnResult)["Candidates"][0]
    #####top1Candidate = ev_vnResults["Id"]
    #####top1Score = ev_vnResults["Score"]*100
    #top1Score = scoreTransform(ev_vnResults["Score"])*100
    match = re.search(r"(.*?jpg):.*?id: (.*?jpg) socre : ([\d.]+)", line)
    if match:
        pass
    else:
        print(line)
    feature_id = match.group(1)
    top1Candidate = match.group(2)
    top1Score = int(float(match.group(3)))

    if 'life100' in feature_id:
        #print(feature_id)
        #print(top1Candidate)
        #print(top1Score)
        if 'license' not in top1Candidate:

            fn += 1
            wrongFZ += 1
            fObj.write('111 feature_id:{} top1Candidate:{} top1Score:{}{}'.format(feature_id, top1Candidate, top1Score, os.linesep))
        elif 'license' in top1Candidate:
            headC = os.path.basename(top1Candidate).split("_")[0]
            headQ = os.path.basename(feature_id).split("_")[0]
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
        else:
            print('else 222222222222')
    if 'black10w' in feature_id:
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
    feaLines = open(featureFile).readlines()

    nObj = open('pr.txt', 'w')

    thresholds = range(70, 101)
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
        nObj.write('threshold:{} wrongRecRate:{} passRate:{}{}'.format(threshold, fpr, tpr, os.linesep))
    nObj.close()
    
    '''
    root@ubuntu:/data/zhouping/faceFeaEval# head -10 result.txt.3516
/nfs/Release/images/cardscence/black10w/9f637c154dc3d3dea740a7a4a0598af9_1.jpg: rect: 98 145 289 332 age: 46 gender: 2 sun_glass: 0(0) mask: 0(0) hat_scarf: 0(0 yaw: 4 pitch: 4 roll: 0 blur: 0.46 match face id: cardscence/black2wNew/369_2.jpg socre : 73.7619
/nfs/Release/images/cardscence/black10w/00ab85307f1f4883d44cd5f1d4a9550b_1.jpg: rect: 107 148 294 329 age: 37 gender: 2 sun_glass: 0(0) mask: 0(0) hat_scarf: 0(0 yaw: 0 pitch: 1 roll: 0 blur: 0.45 match face id: cardscence/black2wNew/3126_3.jpg socre : 76.059
/nfs/Release/images/cardscence/black10w/c73a9fd3170746f0eef3a4618d11b606_1.jpg: rect: 105 146 283 332 age: 54 gender: 1 sun_glass: 0(0) mask: 0(0) hat_scarf: 1(0.862061 yaw: 2 pitch: -1 roll: 0 blur: 0.63 match face id: cardscence/black2wNew/4637_2.jpg socre : 74.7936
/nfs/Release/images/cardscence/black10w/1cb6c6275dfe78470b09df8819c17dc1_1.jpg: rect: 104 146 300 336 age: 23 gender: 1 sun_glass: 0(0) mask: 0(0) hat_scarf: 0(0 yaw: 1 pitch: -2 roll: 1 blur: 0.53 match face id: cardscence/black2wNew/4797_2.jpg socre : 74.5645
/nfs/Release/images/cardscence/black10w/0597ea1ea456d913d44b30014e1afa68_1.jpg: rect: 103 144 284 336 age: 49 gender: 1 sun_glass: 0(0) mask: 0(0) hat_scarf: 0(0 yaw: 0 pitch: 4 roll: 0 blur: 0.62 match face id: cardscence/black2wNew/1566_1.jpg socre : 74.5213
/nfs/Release/images/cardscence/black10w/55a5937ec72e08bf87dbfbcba73ac66e_1.jpg: rect: 105 138 293 321 age: 36 gender: 1 sun_glass: 0(0) mask: 0(0) hat_scarf: 0(0 yaw: 0 pitch: 7 roll: 0 blur: 0.6 match face id: cardscence/black2wNew/4478_0.jpg socre : 74.9623
/nfs/Release/images/cardscence/black10w/dec34e35c102f62616ba0615ba0367eb_1.jpg: rect: 109 150 291 329 age: 34 gender: 1 sun_glass: 0(0) mask: 0(0) hat_scarf: 0(0 yaw: 1 pitch: 0 roll: 2 blur: 0.48 match face id: cardscence/black2wNew/1338_2.jpg socre : 73.3606
/nfs/Release/images/cardscence/black10w/8297a4b257666535b3a0ed18935c57b4_1.jpg: rect: 110 149 303 330 age: 54 gender: 2 sun_glass: 0(0) mask: 0(0) hat_scarf: 0(0 yaw: -4 pitch: 1 roll: 0 blur: 0.62 match face id: cardscence/black2wNew/321_2.jpg socre : 76.6176
/nfs/Release/images/cardscence/black10w/9433b79d1e840c007d821ca13b42a191_1.jpg: rect: 108 138 296 323 age: 33 gender: 1 sun_glass: 0(0) mask: 0(0) hat_scarf: 0(0 yaw: -1 pitch: 9 roll: 0 blur: 0.69 match face id: cardscence/card925/0000025887.jpg socre : 74.191
/nfs/Release/images/cardscence/black10w/052e253ccb3b8efbcf5144d6a81fbda9_1.jpg: rect: 107 144 291 333 age: 42 gender: 2 sun_glass: 0(0) mask: 0(0) hat_scarf: 0(0 yaw: 1 pitch: 1 roll: 0 blur: 0.07 match face id: cardscence/black2wNew/3774_0.jpg socre : 72.0248
    '''
