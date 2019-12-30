#!/usr/bin/python2.7
#!coding=utf-8

import os
import pexpect
import sys
#reload(sys)
#sys.setdefaultencoding("utf-8")

import time
from pylatex import Document, Section, Subsection, Tabular, Math, TikZ, Axis, \
    Plot, Figure, Matrix, Alignat, NoEscape, Command, SubFigure
from pylatex.utils import italic
import csv
import yagmail
import argparse
from perf_util import *
import pdfkit

ap = argparse.ArgumentParser()
ap.add_argument("-lv", "--level", required=True, help = """0:cuda; 1:hisi; 2:hiai; 3:m40""")
ap.add_argument("-gd", "--gpu_id", required=False, default="1", help = "运行测试时所使用的gpu_id,默认使用0号卡")
ap.add_argument("-gt", "--gpu_type", required=False, default="pascal", help = "pascal for default")
ap.add_argument("-pr", "--prompt", required=False, default="#|$", help = "prompt")
ap.add_argument("-es", "--sender", required=False, default="pingzhou@deepglint.com", help = "email")
#ap.add_argument("-er", "--receiver", required=False, default="pingzhou@deepglint.com", help = "email")
ap.add_argument("-er", "--receiver", required=False, default="yufengpan@deepglint.com", help = "email")
#ap.add_argument("-cc", "--cc", required=False, default=["pingzhou@deepglint.com"], help = "cc email")
ap.add_argument("-cc", "--cc", required=False, default=["guoqingjiang@deepglint.com", "ligu@deepglint.com", "hongjiangli@deepglint.com", "wenjialiao@deepglint.com", "zhenxiongchen@deepglint.com", "pingzhou@deepglint.com", "luluqin@deepglint.com", "haoxu@deepglint.com", "linkangchen@deepglint.com", "hailungu@deepglint.com", "dongpingzhang@deepglint.com", "xiaoleizhang@deepglint.com", "qidang@deepglint.com", "zhongjupan@deepglint.com"], help = "cc email")
ap.add_argument("-bh", "--branch", required=False, default="", help = "branch name from CI_COMMIT_REF_NAME")
ap.add_argument("-du", "--duser", required=False, default="", help = "developper from GITLAB_USER_EMAIL")
ap.add_argument("-cm", "--cmess", required=False, default="", help = "check in message from CI_COMMIT_MESSAGE")
ap.add_argument("-mp", "--mpro", required=False, default="", help = "merge project id from CI_MERGE_REQUEST_PROJECT_ID")
ap.add_argument("-mi", "--mid", required=False, default="", help = "merge request id from CI_MERGE_REQUEST_IID")
ap.add_argument("-cn", "--cpn", required=False, default="", help = "merge request id from CI_PROJECT_NAME")
ap.add_argument("-ci", "--cpi", required=False, default="", help = "merge request id from CI_PIPELINE_ID")
ap.add_argument("-vv", "--version", required=False, default="", help = "version")
ap.add_argument("-dm", "--dModel", required=False, default="", help = "debug model")
ap.add_argument("-xm", "--xModel", required=False, default=("PedestrianHelmet", "PedestrianOcclution", "model_group/PlateGraph", "VehicleSideWindow", "TrafficCrosswalk", "VehicleLandmark", "TrafficLineKeypoint", "FaceLandmark_"), help = "models can not automate")

args = vars(ap.parse_args())
level = int(args.get("level"))
gpuId = args.get("gpu_id")
gpuType = args.get("gpu_type")
enable_build_repo = args.get("enable_build_repo")
enable_mini_matrix = args.get("enable_mini_matrix")
enable_functional_test = args.get("enable_functional_test")
enable_run_unittest = args.get("enable_run_unittest")
enable_check_data_md5 = args.get("enable_check_data_md5")
prompt = args.get("prompt")
sender = args.get("sender")
receiver = args.get("receiver")
cc = args.get("cc")
developUser = args.get("duser")
comMessage = args.get('cmess')
mProject = args.get('mpro')
mId = args.get('mid')
cpn = args.get('cpn')
cpi = args.get('cpi')
vegaVersion = args.get("version")
debugModels = args.get("dModel")
exceptModels = args.get("xModel")
branch = args.get("branch")

def main():

    # the the loal session
    session = local(prompt)

    # if gpu_type and gpu_id
    manualGpu(session, gpuType, gpuId, level)

    # get the all the models
    models = getModelLists(level, gpuType)

    # loop through all the models
    qps_summarys = ""
    qps_improve_summarys = ""
    qps_failNum = 0
    qps_totalNum = 0
    qps_failList = []
    qps_passList = []

    cwd = os.getcwd()

    mem_summarys = ""
    mem_improve_summarys = ""
    mem_failNum = 0
    mem_totalNum = 0
    mem_failList = []
    mem_passList = []

    platDict = {0:'Cuda_{}'.format(gpuType), 1:'Hisi', 2:'Hiai', 3:'M40'}
    if level == 0:
        pdfName = gpuType + '_perf'
        csvName = gpuType
    elif level == 1:
        pdfName = 'hisi' + '_perf'
        csvName = 'hisi'
    elif level == 2:
        pdfName = 'hiai' + '_perf'
        csvName = 'hiai'
    elif level == 3:
        pdfName = 'm40' + '_perf'
        csvName = 'm40'

    pdf = Txt2Pdf(pdfName)

    # perf test
    getModelPerf(session=session, level=level, gpuType=gpuType, csvName=cwd + "/" + csvName)
    
    # check the fail or pass
    for model in models:
        if model.startswith(exceptModels):
            continue

        matches  = model.split("_")
        if model.startswith(('FaceFeature', 'ReidNonVehicle', 'ReidPerson', 'ReidVehicleBack', 'ReidVehicleFront')):
            ev_model = matches[0] + "_" + matches[1]
        else:
            ev_model = matches[0]

        if debugModels == "" or ev_model in debugModels:
            pass
        else:
            continue
        print('model:', model)

        qps_summary, qps_improve_summary, mem_summary, mem_improve_summary = summaryPerf(ev_model, csvName)
        if len(qps_summary) != 0:
            qps_failNum += 1
            qps_failList.append(ev_model)
        else:
            qps_passList.append(ev_model)

        qps_summarys += qps_summary
        qps_improve_summarys += qps_improve_summary
        qps_totalNum += 1

        if len(mem_summary) != 0:
            mem_failNum += 1
            mem_failList.append(ev_model)
        else:
            mem_passList.append(ev_model)

        mem_summarys += mem_summary
        mem_improve_summarys += mem_improve_summary
        mem_totalNum += 1

    csvName = cwd + "/" + csvName + '.csv'
    pdf.table(csvName, "allModels", "qps_mem")

    pdf.pdf(pdfName)

    # pass for all perf test
    if level == 2:
        qps_failNum = 0
        qps_passList = qps_passList + qps_failList
        qps_failList = ""

        mem_failNum = 0
        mem_passList = ""
        mem_failList = ""

    if qps_failNum == 0 and mem_failNum == 0:
        status = 'Pass(Perf)'
    else:
        status = 'Fail(Perf)'
 
    # email body
    if mId == "":
        cHead = """
<table border="1" bordercolor="black" cellspacing="0" ID="first">
<tr><td>Branch</td><td>{}</td></tr>
<tr><td>developer</td><td>{}</td></tr>
<tr><td>commit_info</td><td>{}</td></tr>
<tr><td>Trigger</td><td>level:{}</td></tr>
<tr><td>qps_TestResult</td><td>Pass({})<br> Fail({})</td></tr>
<tr><td>mem_TestResult</td><td>Pass({})<br> Fail({})</td></tr>
<tr><td>Pipeline</td><td><a href="https://gitlab.deepglint.com/deepglint/{}/pipelines/{}">pipepline</a></td></tr>
</table><br><br>""".format(vegaVersion, developUser, comMessage, level, qps_passList, qps_failList, mem_passList, mem_failList, cpn, cpi)

        email_status = "[{}] Branch:{} Platform:{}".format(status, vegaVersion, platDict[level])

    else:
        source_branch,target_branch,name,title,description = getMergeInfo(mProject, mId)
        cHead = """
<table border="1" bordercolor="black" cellspacing="0" ID="first">
<tr><td>MergeRequest</td><td>from {} to {}</td></tr>
<tr><td>developer</td><td>{}</td></tr>
<tr><td>MR_title</td><td>{}</td></tr>
<tr><td>MR_destription</td><td>{}</td></tr>
<tr><td>Trigger</td><td>level:{}</td></tr>
<tr><td>qps_TestResult</td><td>Pass({}) <br>Fail({})</td></tr>
<tr><td>mem_TestResult</td><td>Pass({}) <br>Fail({})</td></tr>
<tr><td>Pipeline</td><td><a href="https://gitlab.deepglint.com/deepglint/{}/pipelines/{}">pipepline</a></td></tr>
</table><br><br>""".format(source_branch, target_branch, name, title, description, level, qps_passList, qps_failList, mem_passList, mem_failList, cpn, cpi)

        email_status = "[{}] MergeId:{} Platform:{}".format(status, mId, platDict[level])

    qHead = open('ci/case/eval/css.json').read()
    iqHead = open('ci/case/eval/css.json').read()
    mHead = open('ci/case/eval/css.json').read()
    imHead = open('ci/case/eval/css.json').read()
    #sHead = ""
    cHead = cHead.replace(os.linesep, "")
    qHead += cHead
    iqHead += cHead
    mHead += cHead
    imHead += cHead

    print('qps_failNum:', qps_failNum)
    if qps_failNum != 0:
        qHead += ''' 
<table border="1" bordercolor="black" cellspacing="0" ID="first">
<tr><td>Model</td>
<td><table  border="1" bordercolor="black" cellspacing="0" class="n-bordered" ID="type">
     <tr><td class="td150" >Type</td><td class="td100">Qps(std)</td><td class="td100">Qps(dev)</td><td class="td100">QpsDiff</td></tr></table></td></tr>'''
        qHead += qps_summarys
        qObj = open("qps_decrease_summary.html", "w")
        qObj.write(qHead + "</table>")
        qObj.close()
        pdfkit.from_file('qps_decrease_summary.html','qps_{}_decrease_summary.pdf'.format(pdfName))

        if qps_improve_summarys != "":
            iqHead += ''' 
            <table border="1" bordercolor="black" cellspacing="0" ID="first">
            <tr><td>Model</td>
            <td><table  border="1" bordercolor="black" cellspacing="0" class="n-bordered" ID="type">
             <tr><td class="td150" >Type</td><td class="td100">Qps(std)</td><td class="td100">Qps(dev)</td><td class="td100">QpsDiff</td></tr></table></td></tr>'''
            iqHead += qps_improve_summarys
            qObj = open("qps_increase_summary.html", "w")
            qObj.write(iqHead + "</table>")
            qObj.close()
            pdfkit.from_file('qps_increase_summary.html','qps_{}_increase_summary.pdf'.format(pdfName))
    else:
        pass

    print('mem_failNum:', mem_failNum)
    if mem_failNum != 0:
        mHead += ''' 
<table border="1" bordercolor="black" cellspacing="0" ID="first">
<tr><td>Model</td>
<td><table  border="1" bordercolor="black" cellspacing="0" class="n-bordered" ID="type">
     <tr><td class="td150" >Type</td><td class="td100">Mem(std)</td><td class="td100">Mem(dev)</td><td class="td100">MemDiff</td></tr></table></td></tr>'''
        mHead += mem_summarys
        qObj = open("mem_decrease_summary.html", "w")
        qObj.write(qHead + "</table>")
        qObj.close()
        pdfkit.from_file('mem_decrease_summary.html','mem_{}_decrease_summary.pdf'.format(pdfName))

        if mem_improve_summarys != "":
            imHead += ''' 
            <table border="1" bordercolor="black" cellspacing="0" ID="first">
            <tr><td>Model</td>
            <td><table  border="1" bordercolor="black" cellspacing="0" class="n-bordered" ID="type">
             <tr><td class="td150" >Type</td><td class="td100">Mem(std)</td><td class="td100">Mem(dev)</td><td class="td100">MemDiff</td></tr></table></td></tr>'''
            imHead += mem_improve_summarys
            qObj = open("mem_increase_summary.html", "w")
            qObj.write(imHead + "</table>")
            qObj.close()
            pdfkit.from_file('mem_increase_summary.html','mem_{}_increase_summary.pdf'.format(pdfName))
    else:
        pass

    # get all the pdf 
    pdfList = getAllPdf(cwd)
    sendHtmlEmail(receiver, sender, email_status, qps_summarys, mem_summarys, pdfList, vegaVersion, cc, qps_totalNum, qps_failNum,  mem_totalNum, mem_failNum, cHead)

    # perf test
    #getModelPerf(session=session, level=level, gpuType=gpuType)

if __name__ == '__main__':
    main()
