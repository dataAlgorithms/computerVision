#!/usr/bin/python2.7
#!coding=utf-8

import string
import pexpect
import sys
import re
import os
import time
import os
from pylatex import Document, Section, Subsection, LongTabu, Math, TikZ, Axis, \
    Plot, Figure, Matrix, Alignat, NoEscape, Command, SubFigure
from pylatex.utils import italic
import csv
import yagmail
import logging
import subprocess
import time
import json
import requests
import fnmatch

def getAllPdf(cwd):

    pdfs = []
    for filename in os.listdir(cwd):
        if filename.endswith(".pdf"):
            pdfs.append(filename)
    return pdfs

def qpsMem(qpsname, resultCsv):

    print('resultCsv:', resultCsv)
    modeQpsMemDict = {}

    subject = open(qpsname).read()
    results = re.findall(r"(?i)\| ([0-9a-zA-Z_-]+) \| (\d+) \| (\d+) \| (\d+) \| (\d+) \| (\d+) \| (\d+) \|", subject)
    for result in results:
        model = result[0]
        qps = result[3]

        if model not in modeQpsMemDict:
            modeQpsMemDict[model] = {}
            modeQpsMemDict[model] = {"qps":qps, "mem":"0"}

    subject = open(qpsname).read()
    results = re.findall(r"(?msi)^\| ([0-9a-zA-Z_-]+) \| (\d+) \| (\d+) \| (\d+) \| (\d+) \|$", subject)
    for result in results:
        if len(result[0]) == 0:
            continue
        model = result[0]
        mem = result[4]

        modeQpsMemDict[model]["mem"] = mem

    with open(resultCsv, 'w') as rbj:
        for model in modeQpsMemDict:
            #print(model)
            qps = modeQpsMemDict[model]['qps']
            mem = modeQpsMemDict[model]['mem']
            rbj.write("{},{},{}{}".format(model,qps,mem,os.linesep))

def getModelPerf(session=None, level=0, gpuType="pascal", csvName=None):

    cwd = os.getcwd()
    print('cwd:', cwd)
    if level == 0: # cuda
        jsonFile = 'ci/environment/trt_60_{}.json'.format(gpuType)
        modelFile = 'ci/model_list/trt_60_{}_all.txt'.format(gpuType)
        releasePath = "release/cuda/"
        perlFile = "perf_model_result_trt_60_{}.md".format(gpuType)
    elif level == 1: # nnie
        jsonFile = ""
        modelFile = 'ci/model_list/nnie.txt'
        releasePath = ""
        perlFile = ""
    elif level == 2: # atals
        jsonFile = 'ci/environment/hiai_b896.json'
        modelFile = 'ci/model_list/hiai_b896_all.txt'
        releasePath = "release/hiai/"
        perlFile = "perf_model_result_hiai_b896.md"
    elif level == 3: # m40
        jsonFile = 'ci/environment/trt_60_m40_148.json'
        modelFile = 'ci/model_list/trt_60_maxwell_all.txt'
        releasePath = "release/cuda/"
        perlFile = "perf_model_result_trt_60_m40_148.md"

    retValue = """
    bash {cwd}/ci/run_perf.sh -e {jsonFile} -m {modelFile} -r {releasePath} -o {cwd}
    """.format(jsonFile=jsonFile, modelFile=modelFile, releasePath=releasePath, cwd=cwd)
    print(retValue)
  
    session.sendline("""sed  -i -e 's:.*model_group.*:ci/model_group/PlateGraph.json:' {}""".format(modelFile)) 
    session.expect("#", timeout=None) 
    session.sendline(retValue)
    session.expect("Codec Performance test done", timeout=None)
    session.expect("#", timeout=None)

    time.sleep(10)
    qpsMem(perlFile, "{}.csv".format(csvName))

def getModelLists(level, gpuType='pascal'):

    if level == 0: # cuda
        modelFile = 'ci/model_list/trt_60_{}_all.txt'.format(gpuType)
    elif level == 1: # nnie (hisi)
        modelFile = 'ci/model_list/nnie.txt'
    elif level == 2: # atals (hiai)
        modelFile = 'ci/model_list/hiai_b896_all.txt'
    elif level == 3: # m40
        modelFile = 'ci/model_list/trt_60_maxwell_all.txt'
    
    modelNames = []
    with open(modelFile) as mf:
        for line in mf:
            if line.startswith("#"):
                continue
            match = re.search("([0-9a-zA-_]+)_[if]", line, re.IGNORECASE)
            if match:
                modelName = match.group(1)
                modelNames.append(modelName)
    return modelNames

def manualGpu(session=None, gpuType=None, gpuId=None, level=None):

    if level == 0: # cuda
        session.sendline("""sed  -i -e 's/.*Device.*/    "Device": {},/g' ci/environment/trt_60_{}.json""".format(gpuId, gpuType))
        session.expect("#", timeout=None)
    elif level == 1: # nnie (hisi)
        pass
    elif level == 2: # atlas (hiai)
        session.sendline("""sed  -i -e 's/.*Device.*/    "Device": {},/g' ci/environment/hiai_b896.json""".format(gpuId))
        session.expect("#", timeout=None)
    elif level == 3: # m40
        session.sendline("""sed  -i -e 's/.*Device.*/    "Device": {},/g' ci/environment/trt_60_m40_148.json""".format(gpuId))
        session.expect("#", timeout=None)

    print('manualGpu set ok')

def getMergeInfo(mProject, mId, token="xEhH3tV6G2CxnsSvPXhh"):

    res = requests.get("https://gitlab.deepglint.com/api/v4/projects/{}/merge_requests/{}?private_token={}".format(mProject, mId, token))
    data = json.loads(res.content)

    print('res:', json.dumps(data, indent=1))
    print('mProject:', mProject)
    print('mId:', mId)
    print('token:', token)

    source_branch = data["source_branch"]
    target_branch = data["target_branch"]
    name = data["author"]["name"]
    title = data["title"]
    description = data["description"]
    return source_branch,target_branch,name,title,description
 
class TimeoutError(Exception):
    pass

def excuteCmds(cmds):

    for cmd in cmds:
        excuteCmd(cmd)
 
def excuteCmd(cmd, timeout = 300):
    s = subprocess.Popen(cmd, shell = True, stdout=subprocess.PIPE)

    beginTime = time.time()
    secondsPass = 0

    while True:
        if s.poll() is not None:
            break
        secondsPass = time.time() - beginTime

        if timeout and timeout < secondsPass:
            s.terminate()
            print (u"超时!")
            return False

        time.sleep(0.1)
    return True

def summaryPerf(model=None, ptName=None):


    modelDst = "/mnt/cephfs/testData/ptGt/"
    modelPt = modelDst + "/" + ptName + '.pt'
    modelCsv = '{}.csv'.format(ptName)

    # pr
    qps_summary = """
              <tr><td>{}</td><td>
    """.format(model)
    qps_improve_summary = """
              <tr><td>{}</td><td>
    """.format(model)
    mem_summary = """
              <tr><td>{}</td><td>
    """.format(model)
    mem_improve_summary = """
              <tr><td>{}</td><td>
    """.format(model)

    qflag = 0
    iqflag = 0
    mflag = 0
    imflag = 0

    pDict = {}
    with open(modelPt) as p:
        lines = (line.strip() for line in p)
        for line in lines:
            modelName, qps, mem = line.split(",")
            pDict[modelName] = qps + ':' + mem

    rDict = {}
    with open(modelCsv) as p:
        lines = (line.strip() for line in p)
        for line in lines:
            modelName, qps, mem  = line.split(",")
            rDict[modelName] = qps + ':' + mem

    qps_summary += """<table border="1" bordercolor="black" cellspacing="0" class="n-bordered" ID="carColor">"""
    mem_summary += """<table border="1" bordercolor="black" cellspacing="0" class="n-bordered" ID="carColor">"""

    qps_improve_summary += """<table border="1" bordercolor="black" cellspacing="0" class="n-bordered" ID="carColor">"""
    mem_improve_summary += """<table border="1" bordercolor="black" cellspacing="0" class="n-bordered" ID="carColor">"""

    scene = model
    a_qps, a_mem = map(int, pDict[model].split(":"))
    b_qps, b_mem = map(int, rDict[model].split(":"))

    '''
    if a_qps > b_qps:
        qps_summary += """
           <tr><td class="td150">{}<br></td><td class="td100">{}</td><td class="td100">{}</td><td class="td100"><font color="red">{}</font></td></tr>
              """.format(scene, a_qps, b_qps, round(b_qps-a_qps, 4))
        qflag = 1
    elif a_qps == b_qps:
        pass
    else:
        qps_improve_summary += """
           <tr><td class="td100">{}<br></td><td class="td100">{}</td><td class="td100">{}</td><td class="td100"><font color="green">{}</font></td></tr>
            """.format(scene, a_qps, b_qps, round(b_qps-a_qps, 4))
        iqflag = 1
    '''

    if (a_qps - b_qps) >= 0.2*a_qps:
        qps_summary += """
           <tr><td class="td150">{}<br></td><td class="td100">{}</td><td class="td100">{}</td><td class="td100"><font color="red">{}</font></td></tr>
              """.format(scene, a_qps, b_qps, round(b_qps-a_qps, 4))
        qflag = 1
    elif a_qps - b_qps >= 0 and (a_qps - b_qps) < 0.2*a_qps:
        pass
    elif a_qps - b_qps  < 0 :
        qps_improve_summary += """
           <tr><td class="td100">{}<br></td><td class="td100">{}</td><td class="td100">{}</td><td class="td100"><font color="green">{}</font></td></tr>
            """.format(scene, a_qps, b_qps, round(b_qps-a_qps, 4))
        iqflag = 1

    qps_summary += """</table></td>"""
    qps_improve_summary += """</table></td>"""

    if qflag == 1:
        qps_summary += "</tr>"
    else:
        qps_summary = ""

    if iqflag == 1:
        qps_improve_summary += "</tr>"
    else:
        qps_improve_summary = ""
    '''
    if a_mem < b_mem:
        mem_summary += """
           <tr><td class="td150">{}<br></td><td class="td100">{}</td><td class="td100">{}</td><td class="td100"><font color="red">{}</font></td></tr>
              """.format(scene, a_mem, b_mem, round(b_mem-a_mem, 4))
        mflag = 1
    elif a_mem == b_mem:
        pass
    else:
        mem_improve_summary += """
           <tr><td class="td100">{}<br></td><td class="td100">{}</td><td class="td100">{}</td><td class="td100"><font color="green">{}</font></td></tr>
            """.format(scene, a_mem, b_mem, round(b_mem-a_mem, 4))
        imflag = 1
    '''
    if (b_mem - a_mem) >= 0.2*a_mem:
        mem_summary += """
           <tr><td class="td150">{}<br></td><td class="td100">{}</td><td class="td100">{}</td><td class="td100"><font color="red">{}</font></td></tr>
              """.format(scene, a_mem, b_mem, round(b_mem-a_mem, 4))
        mflag = 1
    elif b_mem - a_mem >=0 and (b_mem - a_mem) < 0.2*a_mem:
        pass
    elif b_mem - a_mem < 0:
        mem_improve_summary += """
           <tr><td class="td100">{}<br></td><td class="td100">{}</td><td class="td100">{}</td><td class="td100"><font color="green">{}</font></td></tr>
            """.format(scene, a_mem, b_mem, round(b_mem-a_mem, 4))
        imflag = 1
    
    mem_summary += """</table></td>"""
    mem_improve_summary += """</table></td>"""

    if mflag == 1:
        mem_summary += "</tr>"
    else:
        mem_summary = ""

    if imflag == 1:
        mem_improve_summary += "</tr>"
    else:
        mem_improve_summary = ""

    return qps_summary, qps_improve_summary, mem_summary, mem_improve_summary

def sendHtmlEmail(receiver, sender, subject, qps_body, mem_body, filename, vegaVersion, cc, qps_totalNum=0, qps_failNum=0, mem_totalNum=0, mem_failNum=0, mHead=""):

    cwd = os.getcwd()

    if qps_body == "" and mem_body == "":
  
        html = """\

@玉峰, @国庆,<br>
<br>
<b>一. 测试版本</b><br>
{}
<br><br>

<b>二. 问题概要</b><br>
qps_totalNum:{}<br>
qps_failNum:{}<br>
mem_totalNum:{}<br>
mem_failNum:{}<br>
{}
<br>

<b>三. 测试详情</b><br>
附件
<br>
<br>

祝好.

<br><br>
-----------------------------------------------<br>
注:<br>
Precision(准确率)=TP÷(TP+FP)，分类器预测出的正样本中，真实正样本的比例<br>
Recall(召回率)=TP÷(TP+FN)，在所有真实正样本中，分类器中能找到多少<br>
F1 Score = P*R/2(P+R)，其中P和R分别为 precision 和 recall<br>
<br>
TP：True Positive，即正确预测出的正样本个数<br>
FP：False Positive，即错误预测出的正样本个数（本来是负样本，被我们预测成了正样本）<br>
TN：True Negative，即正确预测出的负样本个数<br>
FN：False Negative，即错误预测出的负样本个数（本来是正样本，被我们预测成了负样本）<br>

<br>MAE: 平均绝对误差
<br>COS: 欧式距离
<br>LE: 漏检率/误检率
<br>ACC: 平均精确度
<br>PR: 准确率/召回率
<br>-----------------------------------------------<br>
<br><br>
""".format(vegaVersion, qps_totalNum, qps_failNum, mem_totalNum, mem_failNum, mHead)
    elif qps_body != "" and mem_body != "":
        html = """
 <style type="text/css"> 
        #main{
            border: 1px solid blue;
            padding: 10px;
        }

        .bordered, .n-bordered{
            border: 1px solid black;
            border-collapse: collapse;  
        }

        .n-bordered{
            border: none;
        }

        .bordered td, .n-bordered td{
            border: 1px solid black;
        }

        .n-bordered tr:first-child td{
            border-top: none;
        }

        .n-bordered tr:last-child td{
            border-bottom: none;
        }

        .n-bordered tr td:first-child{
            border-left: none;
        }

        .n-bordered tr td:last-child{
            border-right: none;
        }
        .td150{width:250px;overflow:hidden}
        .td100{width:80px;overflow:hidden}

 </style> 
"""

        html += """\

@玉峰, @国庆,<br>
<br>
<b>一. 测试版本</b><br>
{}

<b>二. 问题概要(Qps)</b><br>

<table border="1" bordercolor="black" cellspacing="0" ID="first">
<tr><td>模型</td>
<td><table  border="1" bordercolor="black" cellspacing="0" class="n-bordered" ID="type">
                       <tr><td class="td150" >类型</td><td class="td100">Qps(std)</td><td class="td100">Qps(dev)</td><td class="td100">Qps(diff)</td></tr></table></td></tr>
{}</table>
<br>
<br>

<b>三. 问题概要(Mem)</b><br>

<table border="1" bordercolor="black" cellspacing="0" ID="first">
<tr><td>模型</td>
<td><table  border="1" bordercolor="black" cellspacing="0" class="n-bordered" ID="type">
                       <tr><td class="td150" >类型</td><td class="td100">Mem(std)</td><td class="td100">Mem(dev)</td><td class="td100">Mem(diff)</td></tr></table></td></tr>
{}</table>
<br>
<br>

<b>四. 测试详情</b><br>
附件
<br>
<br>

祝好.

<br><br>
-----------------------------------------------<br>
注:<br>
Precision(准确率)=TP÷(TP+FP)，分类器预测出的正样本中，真实正样本的比例<br>
Recall(召回率)=TP÷(TP+FN)，在所有真实正样本中，分类器中能找到多少<br>
F1 Score = P*R/2(P+R)，其中P和R分别为 precision 和 recall<br>
<br>
TP：True Positive，即正确预测出的正样本个数<br>
FP：False Positive，即错误预测出的正样本个数（本来是负样本，被我们预测成了正样本）<br>
TN：True Negative，即正确预测出的负样本个数<br>
FN：False Negative，即错误预测出的负样本个数（本来是正样本，被我们预测成了负样本）<br>

<br>MAE: 平均绝对误差
<br>COS: 欧式距离
<br>LE: 漏检率/误检率
<br>ACC: 平均精确度
<br>PR: 准确率/召回率
<br>-----------------------------------------------<br>
<br><br>
""".format(mHead, qps_body, mem_body)
    elif qps_body != "" and mem_body == "":
        html = """
 <style type="text/css"> 
        #main{
            border: 1px solid blue;
            padding: 10px;
        }

        .bordered, .n-bordered{
            border: 1px solid black;
            border-collapse: collapse;  
        }

        .n-bordered{
            border: none;
        }

        .bordered td, .n-bordered td{
            border: 1px solid black;
        }

        .n-bordered tr:first-child td{
            border-top: none;
        }

        .n-bordered tr:last-child td{
            border-bottom: none;
        }

        .n-bordered tr td:first-child{
            border-left: none;
        }

        .n-bordered tr td:last-child{
            border-right: none;
        }
        .td150{width:250px;overflow:hidden}
        .td100{width:80px;overflow:hidden}

 </style> 
"""

        html += """\

@玉峰, @国庆,<br>
<br>
<b>一. 测试版本</b><br>
{}

<b>二. 问题概要(Qps)</b><br>

<table border="1" bordercolor="black" cellspacing="0" ID="first">
<tr><td>模型</td>
<td><table  border="1" bordercolor="black" cellspacing="0" class="n-bordered" ID="type">
                       <tr><td class="td150" >类型</td><td class="td100">Qps(std)</td><td class="td100">Qps(dev)</td><td class="td100">Qps(diff)</td></tr></table></td></tr>
{}</table>
<br>
<br>

<b>三. 问题概要(Mem)</b><br>
无
<br>
<br>
<b>四. 测试详情</b><br>
附件
<br>
<br>

祝好.

<br><br>
-----------------------------------------------<br>
注:<br>
Precision(准确率)=TP÷(TP+FP)，分类器预测出的正样本中，真实正样本的比例<br>
Recall(召回率)=TP÷(TP+FN)，在所有真实正样本中，分类器中能找到多少<br>
F1 Score = P*R/2(P+R)，其中P和R分别为 precision 和 recall<br>
<br>
TP：True Positive，即正确预测出的正样本个数<br>
FP：False Positive，即错误预测出的正样本个数（本来是负样本，被我们预测成了正样本）<br>
TN：True Negative，即正确预测出的负样本个数<br>
FN：False Negative，即错误预测出的负样本个数（本来是正样本，被我们预测成了负样本）<br>

<br>MAE: 平均绝对误差
<br>COS: 欧式距离
<br>LE: 漏检率/误检率
<br>ACC: 平均精确度
<br>PR: 准确率/召回率
<br>-----------------------------------------------<br>
<br><br>
""".format(mHead, qps_body)

    elif qps_body == "" and mem_body != "":
        html = """
 <style type="text/css"> 
        #main{
            border: 1px solid blue;
            padding: 10px;
        }

        .bordered, .n-bordered{
            border: 1px solid black;
            border-collapse: collapse;  
        }

        .n-bordered{
            border: none;
        }

        .bordered td, .n-bordered td{
            border: 1px solid black;
        }

        .n-bordered tr:first-child td{
            border-top: none;
        }

        .n-bordered tr:last-child td{
            border-bottom: none;
        }

        .n-bordered tr td:first-child{
            border-left: none;
        }

        .n-bordered tr td:last-child{
            border-right: none;
        }
        .td150{width:250px;overflow:hidden}
        .td100{width:80px;overflow:hidden}

 </style> 
"""

        html += """\

@玉峰, @国庆,<br>
<br>
<b>一. 测试版本</b><br>
{}

<b>二. 问题概要(Qps)</b><br>
无
<br>
<br>

<b>三. 问题概要(Mem)</b><br>

<table border="1" bordercolor="black" cellspacing="0" ID="first">
<tr><td>模型</td>
<td><table  border="1" bordercolor="black" cellspacing="0" class="n-bordered" ID="type">
                       <tr><td class="td150" >类型</td><td class="td100">Mem(std)</td><td class="td100">Mem(dev)</td><td class="td100">Mem(diff)</td></tr></table></td></tr>
{}</table>
<br>
<br>

<b>四. 测试详情</b><br>
附件
<br>
<br>

祝好.

<br><br>
-----------------------------------------------<br>
注:<br>
Precision(准确率)=TP÷(TP+FP)，分类器预测出的正样本中，真实正样本的比例<br>
Recall(召回率)=TP÷(TP+FN)，在所有真实正样本中，分类器中能找到多少<br>
F1 Score = P*R/2(P+R)，其中P和R分别为 precision 和 recall<br>
<br>
TP：True Positive，即正确预测出的正样本个数<br>
FP：False Positive，即错误预测出的正样本个数（本来是负样本，被我们预测成了正样本）<br>
TN：True Negative，即正确预测出的负样本个数<br>
FN：False Negative，即错误预测出的负样本个数（本来是正样本，被我们预测成了负样本）<br>

<br>MAE: 平均绝对误差
<br>COS: 欧式距离
<br>LE: 漏检率/误检率
<br>ACC: 平均精确度
<br>PR: 准确率/召回率
<br>-----------------------------------------------<br>
<br><br>
""".format(mHead, mem_body)

    print('filename:', filename)
    while True:
        try:
            print('yagmail:', yagmail.__version__)
            yag = yagmail.SMTP(user=sender,password='6LfNHZs7uHhBBABY',host='smtp.exmail.qq.com')

            html = html.replace(os.linesep, "")

            body = 'This is obviously the body'

            #attachments = [filename, cwd + "/ssh.txt", "/builds/deepglint/vega/script/vegaAuto/eval/carColor/carColor.csv"]
            attachments = filename
            print('attachments:', attachments)
            yag.send(to=receiver, subject=subject, contents=[html], attachments=attachments, cc=cc)
            break
        except  Exception as e:
            print('e:', e)
            time.sleep(60)
            print('Email server can not access to!')


class Txt2Pdf:
    def __init__(self, version=None):
        self.doc = Document('basic')
        self.doc.preamble.append(Command('title', 'Vega Evaluation'))
        self.doc.preamble.append(Command('author', 'vega {}'.format(version)))
        self.doc.preamble.append(Command('date', NoEscape(r'\today')))
        self.doc.append(NoEscape(r'\maketitle'))

    def table(self, csvFile=None, cType=None, pType=None, level=None):
        with self.doc.create(Section(cType)):
            self.doc.append('{} evaluation'.format(cType))
            with self.doc.create(Subsection('Qps/Mem detail')):

                with self.doc.create(LongTabu('|c|c|c|')) as table:
                    table.add_hline()
                    table.add_row(("Model", "Qps", "Mem"))
                    table.add_hline()

                    with open(csvFile) as f:
                        f_csv = csv.reader(f)
                        for row in f_csv:
                            m,q,e= row
                            if level == 2:
                                e = '-'
                            table.add_row((m,q,e))
                            table.add_hline()
    def pdf(self, pName):
        self.doc.generate_pdf(pName, clean_tex=False)

def local(prompt='#'):

    session = pexpect.spawn("/bin/bash", timeout=None, encoding='utf-8')

    fout = open('ssh.txt','w')
    session.logfile = fout
    session.expect(prompt)
    #session.read()
    return session

def getVersion(prompt='#'):

    cwd = os.getcwd()

    session = pexpect.spawn("git branch {}".format(cwd), timeout=None)
    subject = session.readline()
    version = re.search("at.*?([a-zA-Z0-9_-]+)", subject).group(1)
    return version

def vegaUnitEmail(receiver, sender, subject, body):

    yag = yagmail.SMTP(user='pingzhou@deepglint.com',password='6LfNHZs7uHhBBABY',host='smtp.exmail.qq.com')

    html = """\
@玉峰, @国庆, @谷力<br>
<b>一. 测试版本</b><br>
{}""".format(body)

    #html = html.replace(os.linesep, "")

    yag.send(
        to=receiver,
        subject=subject,
        contents=[html]
    )

def sendCmd(session=None, cmds=None, prompt='#'):

    for cmd in cmds:
        session.sendline(cmd)
        if 'run_case.sh' in cmd:
            match = re.search(r"grep -i ([0-9a-zA-Z_]+)", cmd).group(1)
            matches = match.split("_")
            if match.startswith(('FaceFeature', 'ReidNonVehicle', 'ReidPerson', 'ReidVehicleBack', 'ReidVehicleFront')):
                match = matches[0] + "_" + matches[1]
            else:
                match = matches[0]

            expectMatch = "{} test is finished".format(match)
            ret = session.expect([expectMatch, 'Check failed: error', 'error code is'], timeout=None)
            print('ret:', ret)
            if ret == 0:
                pass
            elif ret == 1 or ret == 2:
                buffer = session.before[len(session.before)-2000:]
                vegaUnitEmail("pingzhou@deepglint.com", "pingzhou@deepglint.com", "Newcuda test Fail", buffer)
                sys.exit(1)
        else:
            #session.expect(prompt, timeout=None)
            session.expect(cmd[:5], timeout=None)

        session.expect(prompt, timeout=None)
        session.buffer = "" 

    session.sendline('ls')
    session.expect('ls')
    session.expect(prompt)

def generateCmd(level=0, modelName=None, gpuType="pascal"):

    cwd = os.getcwd()

    if level == 0: # cuda
        jsonFile = 'ci/environment/trt_60_{}.json'.format(gpuType)
        modelFile = 'ci/model_list/trt_60_{}_all.txt'.format(gpuType)
        releasePath = "release/cuda/"
    elif level == 1: # nnie
        jsonFile = ""
        modelFile = 'ci/model_list/nnie.txt'
        releasePath = ""
    elif level == 2: # atals
        jsonFile = 'ci/environment/hiai_b896.json'
        modelFile = 'ci/model_list/hiai_b896_all.txt'
        releasePath = "release/hiai/"
    elif level == 3: # m40
        jsonFile = 'ci/environment/trt_60_m40_148.json'
        modelFile = 'ci/model_list/trt_60_maxwell_all.txt'
        releasePath = "release/cuda/"

    retValue = """
    bash {cwd}/ci/run_case.sh -e {jsonFile} -m `grep -i {modelName} {modelFile}` -r {releasePath} -t precision -o {cwd}
    """.format(jsonFile=jsonFile, modelName=modelName, modelFile=modelFile, releasePath=releasePath, cwd=cwd)
    print(retValue)
    return retValue

def evalCmd(level=None, model=None):

    modelDst = "ci/case/eval/utils/" + model
    modelPy = "ci/case/eval/utils/" + model + "/" + model + '.py'
    #modelGt = "ci/case/eval/utils/" + model + "/" + model + '.gt'
    modelCsv = "ci/case/eval/utils/" + model + "/" + model + '.csv'

    for result in sorted(os.listdir(".")):
        if not result.startswith("result."):
            continue
       
        match = re.search(r"result\.(.*)", result, re.IGNORECASE).group(1)
        matches = match.split("_")

        if match.startswith(('FaceFeature', 'ReidNonVehicle', 'ReidPerson', 'ReidVehicleBack', 'ReidVehicleFront')):
            match = matches[0] + "_" + matches[1]
            modelGt = "/mnt/cephfs/testData/ptGt" + "/" + match + '.gt'
            modelRt = "ci/case/eval/utils/" + model + "/" + match + '.rt'
            os.system("cp result.{match}* {modelDst}/{match}.rt;python2 {modelPy} {modelGt} {modelRt} {modelCsv} {match}".format(match=match, modelDst=modelDst, model=model, modelPy=modelPy, modelRt=modelRt, modelGt=modelGt, modelCsv=modelCsv))
        elif match.startswith(('NonMotorAttribute', 'PedestrianAttribute')):
            match = matches[0]
            modelGt = "/mnt/cephfs/testData/ptGt" + "/" + match + '.gt'
            modelRt = "ci/case/eval/utils/" + model + "/" + match + '.rt'
            time.sleep(2)
            print("cp result.{match}* {modelDst}/{match}.rt;python2 {modelPy} {modelGt} {modelRt} {modelCsv} {match}".format(match=match, modelDst=modelDst, model=model, modelPy=modelPy, modelRt=modelRt, modelGt=modelGt, modelCsv=modelCsv))
            os.system("cp result.{match}* {modelDst}/{match}.rt;python2 {modelPy} {modelGt} {modelRt} {modelCsv} {match}".format(match=match, modelDst=modelDst, model=model, modelPy=modelPy, modelRt=modelRt, modelGt=modelGt, modelCsv=modelCsv))
        else:
            match = matches[0]
            modelGt = "/mnt/cephfs/testData/ptGt" + "/" + match + '.gt'
            modelRt = "ci/case/eval/utils/" + model + "/" + match + '.rt'
            time.sleep(2)
            print("cp result.{match}* {modelDst}/{match}.rt;python {modelPy} {modelGt} {modelRt} {modelCsv} {match}".format(match=match, modelDst=modelDst, model=model, modelPy=modelPy, modelRt=modelRt, modelGt=modelGt, modelCsv=modelCsv))
            os.system("cp result.{match}* {modelDst}/{match}.rt;python {modelPy} {modelGt} {modelRt} {modelCsv} {match}".format(match=match, modelDst=modelDst, model=model, modelPy=modelPy, modelRt=modelRt, modelGt=modelGt, modelCsv=modelCsv))

    os.system("rm result.*")
