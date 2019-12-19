#!/usr/bin/env python
#!coding=utf-8

import time
import re
import os
import string
import requests
import json
#import psycopg2

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

import sys
#reload(sys)
#sys.setdefaultencoding('utf-8')

import json
import uuid

import numpy as np
import base64
from optparse import OptionParser
import math
import sys

def array2feature(array):
    aList = [eval(i) for i in array.split()]
    featureFloat = np.array(aList,dtype=np.float32)
    array = np.array(featureFloat)
    array = array/math.sqrt(2)
    feature = base64.b64encode(array)
    #print 'feature:', feature

    return feature

import matplotlib
import numpy as np
from numpy import *
import matplotlib as mpl
mpl.use('Agg')
from matplotlib import pyplot as plt
from matplotlib.font_manager import FontProperties

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
    #print '::Do the repo add'
    add_source = {"Context":{},"Repo":{"Operation":1,"RepoId":repoId,"Level":3,"FeatureLen":int(featureLen),"Capacity":int(capacity),"Params":{"GPUThreads":"%s" % gPUThreads}}}
    repo.addRepo(add_source)

# Feature class of Ranker2
class Ranker2Feature:
    def __init__(self, url):
        self.url = url

    def addFeature(self, source):
        resp_dict, _ret = post_request(self.url, source)
        #print '::Add feature result is as follow!'
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
def vNRanker(rankUrl="http://192.168.2.16:6501/rank", repoId=None, location=None, featureStr=None, maxCandiates=10):

     source = {"Params":{"RepoId":repoId,"Normalization":"false","MaxCandidates":str(maxCandiates)},"ObjectFeature":{"Feature":featureStr},"Context":{"SessionId":"test123"}}

     resp_dict,ret = post_request(rankUrl,source)
     #print json.dumps(resp_dict, indent=1)
     return json.dumps(resp_dict, indent=1)
     try:
         return resp_dict["Candidates"][0]["Score"],resp_dict["Candidates"][0]["Id"]
     except:
         return None

def _1v1RankerFromFile(rankerUrl=None, repoId=None, featureFile=None, topNum=200,topType=1, threshold=1, pType='carback'):

    # get total for each type
    totalNum = {}
    with open(featureFile) as f:
        for line in f:
            subject = line.strip()
            #line = line.split(" ")
            #fea_id = line[0]
            #fea_id = re.search(r"/([0-9_]+)\.jpg", fea_id).group(1)
            #fea_id = re.split("[-_\.]", fea_id)[0]
            fea_id = re.search("{}/(.*?)/".format(pType), subject, re.DOTALL | re.IGNORECASE).group(1)
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

    nobj = open("%s.result" % topType, "w+")

    topHitList = []
    for line in allFeatures:
        line = line.strip("\n")
        feature_line = line.split(" ")
        #feature_id = feature_line[0].encode("utf-8")
        #feature_id = re.search(r"/([0-9_]+)\.jpg", feature_id).group(1)
        feature_id = re.search("{}/(.*?)/".format(pType), line, re.DOTALL | re.IGNORECASE).group(1)
        feature_str = feature_line[1]
        #print ( "[%s] [%s] [%s]" % (feature_line, feature_id, feature_str))

        if feature_str == str(None):
            continue

        vnResult = vNRanker(rankUrl=rankerUrl, repoId=repoId, location=0, featureStr=feature_str, maxCandiates=int(topNum))
        ev_vnResults = eval(vnResult)["Candidates"]

        len_vnResults = len(ev_vnResults)

        topHitNum = 0
        topHitNot = 0

        topList = []
        scoreList = []
        feature_id_bak = feature_line[0]
        #feature_id = re.split("[-_\.]", feature_id)[0]

        for i in range(len_vnResults):
            #top_id = re.split("[-_\.]", ev_vnResults[i]["Id"])[0]
            subject = ev_vnResults[i]["Id"]
            top_id_match = re.search("{}/(.*?)/".format(pType), subject, re.DOTALL | re.IGNORECASE)
            score_id = ev_vnResults[i]["Score"]

            if top_id_match:
                top_id = top_id_match.group(1)
            else:
                top_id = 'distrib'
            #print('top_id:{} fea_id:{}'.format(top_id, feature_id))
            if top_id != feature_id:
                topHitNot += 1
            else:
                if float(score_id) >= threshold:
                    continue
                topHitList.append(ev_vnResults[i]["Id"])
                topHitNum += 1

            topList.append(ev_vnResults[i]["Id"])
            scoreList.append(score_id)

            if topType == topHitNot:
                break

        if totalNum[feature_id] == 1:
            continue

        child_hit = 1.0*(topHitNum-1)/(totalNum[feature_id]-1)

        topN_hit += 1.0*(topHitNum-1)/(totalNum[feature_id]-1)
        topN_total += 1

        topHitNums += topHitNum-1
        topNums += totalNum[feature_id]
        nobj.write('fea:%s topList:%s topHitNum:%s totalNum:%s hit:%s%s' % (feature_id_bak, topList, topHitNum,totalNum[feature_id], child_hit, os.linesep))
        #print('fea:%s topList:%s topHitNum:%s totalNum:%s hit:%s' % (feature_id_bak, topList, topHitNum,totalNum[feature_id], child_hit))

    tophit = 1.0*topN_hit/topN_total

    unique_hitHistList = set(topHitList)

    #print('unique_hitHistList:', unique_hitHistList)
    print("topType:top{} tophitRate:{} totalNum(threshold:{}):{}".format(topType, tophit,threshold,len(unique_hitHistList)))

    nobj.write("%s" % tophit)
    nobj.close()
    #print("top: ", 1.0*topHitNums/topNums)

if __name__ == "__main__":
    # top1 or top10
    topType = int(sys.argv[1])
    scoreThreshold = float(sys.argv[2])
    featureFile = sys.argv[3]
    repoId = sys.argv[4]
    pType = sys.argv[5]

    # ranker
    _1v1RankerFromFile(rankerUrl="http://192.168.2.16:8010/rank", repoId=repoId, featureFile=featureFile, topType=topType, threshold=scoreThreshold, pType=pType)
