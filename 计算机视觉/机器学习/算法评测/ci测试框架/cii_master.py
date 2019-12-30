#!/usr/bin/python3
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
from cii_util import *
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
    summarys = ""
    improve_summarys = ""
    failNum = 0
    totalNum = 0
    pdf = Txt2Pdf(vegaVersion)
    failList = []
    passList = []
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

        cmds = generateCmd(level, model, gpuType)
        cmds = [cmd.strip() for cmd in  cmds.split(os.linesep)]
        sendCmd(session, cmds)
        evalCmd(level, ev_model)
        summary, improve_summary = summaryGet(ev_model)
        if len(summary) != 0:
            failNum += 1
            failList.append(ev_model)
        else:
            passList.append(ev_model)

        summarys += summary
        improve_summarys += improve_summary
        csvName = "ci/case/eval/utils/" + ev_model + "/" + ev_model + '.csv'

        pdf.table(csvName, model, "pr")

        totalNum += 1

    platDict = {0:'Cuda_{}'.format(gpuType), 1:'Hisi', 2:'Hiai', 3:'M40'}
    if level == 0:
        pdfName = gpuType
    elif level == 1:
        pdfName = 'hisi'
    elif level == 2:
        pdfName = 'hiai'
    elif level == 3:
        pdfName = 'm40'

    pdf.pdf(pdfName)

    if failNum != 0:
        status = 'Fail'
    else:
        status = 'Pass' 
    # email body
    if mId == "":
        mHead = """
<table border="1" bordercolor="black" cellspacing="0" ID="first">
<tr><td>Branch</td><td>{}</td></tr>
<tr><td>developer</td><td>{}</td></tr>
<tr><td>commit_info</td><td>{}</td></tr>
<tr><td>Trigger</td><td>level:{}</td></tr>
<tr><td>TestResult</td><td>Pass({})<br> Fail({})</td></tr>
<tr><td>Pipeline</td><td><a href="https://gitlab.deepglint.com/deepglint/{}/pipelines/{}">pipepline</a></td></tr>
</table><br><br>""".format(vegaVersion, developUser, comMessage, level, passList, failList, cpn, cpi)

        email_status = "[{}] Branch:{} Platform:{}".format(status, vegaVersion, platDict[level])

    else:
        source_branch,target_branch,name,title,description = getMergeInfo(mProject, mId)
        mHead = """
<table border="1" bordercolor="black" cellspacing="0" ID="first">
<tr><td>MergeRequest</td><td>from {} to {}</td></tr>
<tr><td>developer</td><td>{}</td></tr>
<tr><td>MR_title</td><td>{}</td></tr>
<tr><td>MR_destription</td><td>{}</td></tr>
<tr><td>Trigger</td><td>level:{}</td></tr>
<tr><td>TestResult</td><td>Pass({}) <br>Fail({})</td></tr>
<tr><td>Pipeline</td><td><a href="https://gitlab.deepglint.com/deepglint/{}/pipelines/{}">pipepline</a></td></tr>
</table><br><br>""".format(source_branch, target_branch, name, title, description, level, passList, failList, cpn, cpi)

        email_status = "[{}] MergeId:{} Platform:{}".format(status, mId, platDict[level])

    sHead = open('ci/case/eval/css.json').read()
    iHead = open('ci/case/eval/css.json').read()
    #sHead = ""
    mHead = mHead.replace(os.linesep, "")
    sHead += mHead
    iHead += mHead

    if failNum != 0:
        sHead += ''' 
<table border="1" bordercolor="black" cellspacing="0" ID="first">
<tr><td>Model</td>
<td><table  border="1" bordercolor="black" cellspacing="0" class="n-bordered" ID="type">
                       <tr><td class="td150" >Type</td><td class="td100">P(dev)</td><td class="td100">P(master)</td><td class="td100">R(dev)</td><td class="td100">R(master)</td><td class="td100">F1(dev)</td><td class="td100">F1(master)</td><td class="td100">F1 Diff</td></tr></table></td></tr>'''
        sHead += summarys
        sObj = open("decrease_summary.html", "w")
        sObj.write(sHead + "</table>")
        sObj.close()
        pdfkit.from_file('decrease_summary.html','{}_decrease_summary.pdf'.format(pdfName))

        if improve_summarys == "":
            sendHtmlEmail(receiver, sender, email_status, summarys, ["{}.pdf".format(pdfName), "{}_decrease_summary.pdf".format(pdfName)], vegaVersion, cc, totalNum, failNum, mHead)
        else:

            iHead += ''' 
            <table border="1" bordercolor="black" cellspacing="0" ID="first">
            <tr><td>Model</td>
            <td><table  border="1" bordercolor="black" cellspacing="0" class="n-bordered" ID="type">
            <tr><td class="td150" >Type</td><td class="td100">P(dev)</td><td class="td100">P(master)</td><td class="td100">R(dev)</td><td class="td100">R(master)</td><td class="td100">F1(dev)</td><td class="td100">F1(master)</td><td class="td100">F1 Diff</td></tr></table></td></tr>'''
            iHead += improve_summarys
            sObj = open("increase_summary.html", "w")
            sObj.write(iHead + "</table>")
            sObj.close()
            pdfkit.from_file('increase_summary.html','{}_increase_summary.pdf'.format(pdfName))
            sendHtmlEmail(receiver, sender, email_status, summarys, ["{}.pdf".format(pdfName), "{}_decrease_summary.pdf".format(pdfName), "{}_increase_summary.pdf".format(pdfName)], vegaVersion, cc, totalNum, failNum, mHead)
        sys.exit(1)
    else:
        if improve_summarys == "":
            sendHtmlEmail(receiver, sender, email_status, summarys, ["{}.pdf".format(pdfName)], vegaVersion, cc, totalNum, failNum, mHead)
        else:
            iHead += ''' 
            <table border="1" bordercolor="black" cellspacing="0" ID="first">
            <tr><td>Model</td>
            <td><table  border="1" bordercolor="black" cellspacing="0" class="n-bordered" ID="type">
            <tr><td class="td150" >Type</td><td class="td100">P(dev)</td><td class="td100">P(master)</td><td class="td100">R(dev)</td><td class="td100">R(master)</td><td class="td100">F1(dev)</td><td class="td100">F1(master)</td><td class="td100">F1 Diff</td></tr></table></td></tr>'''
            iHead += improve_summarys
            sObj = open("increase_summary.html", "w")
            sObj.write(iHead + "</table>")
            sObj.close()
            pdfkit.from_file('increase_summary.html','{}_increase_summary.pdf'.format(pdfName))
            sendHtmlEmail(receiver, sender, email_status, summarys, ["{}.pdf".format(pdfName), "{}_increase_summary.pdf".format(pdfName)], vegaVersion, cc, totalNum, failNum, mHead)
        os.system("rm ssh.txt")
        sys.exit(0)

if __name__ == '__main__':
    main()
