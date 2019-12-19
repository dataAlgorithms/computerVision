#!/usr/bin/env python
#!coding=utf-8

import time
import re
import os
import string
import requests
import json

import numpy as np
import base64
from optparse import OptionParser
import math


import numpy as np
import math
import base64
import random

import codecs
from collections import defaultdict
from fnmatch import fnmatch

import urllib

import sys
reload(sys)
sys.setdefaultencoding('utf-8')

import matplotlib
import numpy as np
from numpy import *
import matplotlib as mpl
mpl.use('Agg')
from matplotlib import pyplot as plt
from matplotlib.font_manager import FontProperties

import matplotlib as mpl
import matplotlib.pyplot as plt
from IPython.display import Image
import numpy as np
import cv2

repoId=sys.argv[1]
csvFile = sys.argv[2]
rankIps = sys.argv[3]
rankPorts = sys.argv[4]

totalNums = {}
with open(csvFile) as f:
    for line in f:
        line = line.strip()
        line = line.split(" ")
        fea_id = line[0]
        fea_id = re.search(r"/([0-9_]+)\.jpg", fea_id).group(1)
        fea_id = re.split("[-_\.]", fea_id)[0]

        if fea_id in totalNums:
            totalNums[fea_id] += 1
        else:
            totalNums[fea_id] = 1

def drawSd(picList, rateList, confList, costList, filename, topNum):

    fig = plt.figure(figsize=(10, 6), dpi=128)
    ax = plt.subplot(111)

    p1 = ax.scatter(picList, rateList,marker = '*',color = 'r',label='1',s=10)
    p2 = ax.scatter(picList, confList,marker = 'o',color = 'g',label='1',s=10)
    p3 = ax.scatter(picList, costList,marker = '+',color = 'b',label='1',s=10)

    plt.xlabel(u'picture')
    plt.ylabel(u'top%s' % topNum)
    ax.legend((p1,p2,p3), (u'recallRate', u'minConfidence', u'avgCost'), loc=2)
    #plt.show()
    fig.savefig('%s_top%s.png' % (filename, topNum), bbox_inches='tight')

def post_request(url, source):
    jsource = json.dumps(source)

    #print 'jsource:', jsource
    resp = requests.post(url, data = jsource)
    if resp.content == "":
        return None,resp.status_code
    else:
        rdict = json.loads(resp.content)
        return rdict,resp.status_code

def get_request(url):
    resp = requests.get(url)
    if resp.content == "":
        return None,resp.status_code
    else:
        rdict = json.loads(resp.content)
        return rdict,resp.status_code

def del_request(url):
    resp = requests.delete(url)
    if resp.content == "":
        return None,resp.status_code
    else:
        rdict = json.loads(resp.content)
        return rdict,resp.status_code

def put_request(url, source):
    jsource = json.dumps(source)
    resp = requests.put(url, data = jsource)
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
        print 'resp: ', json.dumps(resp_dict, indent=1)

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
def vNRanker(rankUrl="http://192.168.2.16:6501/rank", repoId=None, location=None, featureStr=None, maxCandiates=10):

     source = {"Params":{"RepoId":repoId,"Normalization":"false","MaxCandidates":str(maxCandiates)},"ObjectFeature":{"Feature":featureStr},"Context":{"SessionId":"test123"}}

     resp_dict,ret = post_request(rankUrl,source)
     #print json.dumps(resp_dict, indent=1)
     return json.dumps(resp_dict, indent=1)
     try:
         return resp_dict["Candidates"][0]["Score"],resp_dict["Candidates"][0]["Id"]
     except:
         return Non

def drawQueryGalary(galaryList, scoreList):

    newObj = open("newgalary.txt", "a+")
    scoreObj = open("score.txt", "a+")

    outputname = os.path.basename(galaryList[0]).split(".")[0] + "_20galary.jpg"

    newGalaryList = []
    for galary in galaryList:
        if galary.startswith("file://"):
            galary = galary.lstrip("file:")

            newGalaryList.append(galary)

        elif galary.startswith("http:"):
            if galary.endswith(".jpg"):
                basename = os.path.basename(galary)
            else:
                basename = os.path.basename(galary) + ".jpg"
            try:

                if not os.path.exists(basename):
                    urllib.urlretrieve(galary.strip(), basename)

                galary = os.getcwd() + "/" + basename

                newGalaryList.append(galary)
                print('galary:', galary)
            except:
                newGalaryList.append(None)
        else:
            if 'black' in galary:
                galary = "images/carfront/" + os.path.basename(galary)
            elif 'vehicle200w' in galary:
                galary = "images/carfront200w/" + os.path.basename(galary)
            else:
                galary = "images/carfront/" + os.path.basename(galary) + ".jpg"
            #galary = "/data/zhouping/algoSdk/code/dgreid/degreid_evaluate/frommatrix/black" + "/" +  galary
            #galary = "/data/zhouping/scripts/reidPc/black/images/carback" + "/" + galary + ".jpg"
            #galary = "/data/zhouping/scripts/reidPc/black/newblack/person/hard" + "/" + re.split("[-_\.]", galary)[0] + "/" + galary + ".jpg"
            #galary = "/data/zhouping/scripts/reidPc/black/images/car" + "/" + re.split("[-_\.]", galary)[0] + "/" + galary + ".jpg"
            #galary = "/data/zhouping/algoSdk/code/dgreid/degreid_evaluate/frommatrix/scripts/images/carfront" + "/" + re.split("[-_\.]", galary)[0] + "/" + galary + ".jpg"
            newGalaryList.append(galary)
            print('galary:', galary)

    #print('newGalaryList:', newGalaryList)
    newObj.write("%s" % newGalaryList)
    newObj.write(os.linesep)
    newObj.close()

    newScoreList = [item for item in scoreList if item is not None]

    scoreObj.write("%s" % newScoreList)
    scoreObj.write(os.linesep)
    scoreObj.close()

    plt.figure()
    fig, ax = plt.subplots(3, 7,figsize=(20, 20))
    fig.suptitle('totalNum:' + str(totalNums[os.path.basename(newGalaryList[0]).split(".jpg")[0].split("_")[0]]), fontsize=20)
    #ax.set(title='totalNum:' + str(os.path.basename(newGalaryList[0]).split(".jpg")[0].split("_")[0]))
    #ax.set_title('totalNum:' + str(os.path.basename(newGalaryList[0]).split(".jpg")[0].split("_")[0]),fontsize=12,color='r')
    #fig.subplots_adjust(hspace=0, wspace=0)
    #fig.tight_layout()

    for i in range(3):
        for j in range(7):

            qBasename = os.path.basename(newGalaryList[0]).split(".jpg")[0].split("_")[0]
            index = 7*i+j
            img = cv2.imread(newGalaryList[index])
            basename = os.path.basename(newGalaryList[index]).split(".jpg")[0].split("_")[0]
            score = scoreList[index]
            if img is None:
                continue

            b,g,r= cv2.split(img)
            img2 = img[:,:,::-1]
            ax[i, j].xaxis.set_major_locator(plt.NullLocator())
            ax[i, j].yaxis.set_major_locator(plt.NullLocator())

            if score is None:
                ax[i, j].text(0, 0, 'Query')
            else:
                score = round(float(score), 3)
                if qBasename != basename:
                    ax[i, j].text(0, 0, str(score), fontsize=20)
                else:
                    ax[i, j].text(0, 0, str(score)+"_"+"y", fontsize=20)
            ax[i,j].imshow(img2,cmap="bone")

    plt.savefig("output/%s" % outputname)

    #os.system("rm *.jpg")

def _1v1RankerFromFile(rankerUrl=None, repoId=None, featureFile=None, topNum=20,topType=1):

    # get total for each type
    totalNum = {}
    with open(featureFile) as f:
        for line in f:
            line = line.strip()
            line = line.split(" ")
            fea_id = line[0]
            fea_id = re.search(r"/([0-9_]+)\.jpg", fea_id).group(1)
            fea_id = re.split("[-_\.]", fea_id)[0]
            if fea_id in totalNum:
                totalNum[fea_id] += 1
            else:
                totalNum[fea_id] = 1

    # Fetch all features
    fObj = open(featureFile)
    allFeatures = fObj.readlines()
    fObj.close()

    # Do through all features
    topIdScoreDict = {}

    topN_hit = 0
    topN_total = 0

    topHitNums = 0
    topNums = 0

    for line in allFeatures:

        scoreList = []
        queryGalaryList = []
        line = line.strip("\n")
        feature_line = line.split(" ")
        feature_id = feature_line[0].encode("utf-8")

        picture_url = feature_id

        queryGalaryList.append(picture_url)
        scoreList.append(None)

        feature_id = re.search(r"/([0-9_]+)\.jpg", feature_id).group(1)
        feature_str = feature_line[1]

        if feature_str == str(None):
            continue

        vnResult = vNRanker(rankUrl=rankerUrl, repoId=repoId, location=0, featureStr=feature_str, maxCandiates=int(topNum))
        ev_vnResults = eval(vnResult)["Candidates"]

        len_vnResults = len(ev_vnResults)
        for ev in range(len_vnResults):
            galary = ev_vnResults[ev]["Id"]
            score = ev_vnResults[ev]["Score"]

            queryGalaryList.append(galary)
            scoreList.append(score)

        print('qeuryGalaryList:', queryGalaryList)
        print('scoreList:', scoreList)
        drawQueryGalary(queryGalaryList, scoreList)

    #print('qeuryGalaryList:', queryGalaryList)

if __name__ == "__main__":
    # remove
    try:
        os.system("rm newgalary.txt")
        os.system("rm score.txt")
    except:
        pass

    # ranker
    _1v1RankerFromFile(rankerUrl="http://{}:{}/rank".format(rankIps, rankPorts), repoId=repoId, featureFile=csvFile)
