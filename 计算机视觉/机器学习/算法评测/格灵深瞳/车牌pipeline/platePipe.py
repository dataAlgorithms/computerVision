import sys
import os
import re
import argparse
import magic # pip install python-magic
import json
import fnmatch
import pexpect
import cv2
from PIL import Image, ImageDraw, ImageFont
import numpy as np

fontpath = "/mnt/cephfs/testData/ptGt/NotoSansCJK-Black.ttc" #https://pan.baidu.com/s/1y8i8fEg5Hk2UQo6G6dz_BA
pfont = ImageFont.truetype(fontpath, 20)

'''
1.deal with image/directory/list
obj:
create the list (including absolute path of image)
2.modify the vehicle detect json
obj:
use the list above as the list of VehicleDetector json
3.get the docker session
obj:
assign the session to the variable:docSess
4.run the case of VehicleDetector 
obj:
get the detect result
5.crop the image according the detect result (only vehicle)
obj:
get the new image (small vehicle image)
6. make the list with the above new image
obj:
create the list for the platedetect
7. modify the PlateDetect json
obj:
use the list above as the list of PlateDetector json
8. run the case of PlateDetector
obj:
get the result of plateDetector
9. modify the result of plateDetector as the input list of plateGraph
10.modify the plateColor/plateChar/PlateRcity list(only one line)
11.run the plateGraph
obj:
get the result of plateGraph
12.virtualize the result of plategraph 
'''
colorDict = {
    "-1": "Unknown",
    "1": "Blue",
    "2": "Yellow",
    "3": "White",
    "4": "Black",
    "5": "Green",
    "6": "YellowGreen",
    "7": "HalfGreen"
}

ap = argparse.ArgumentParser()
ap.add_argument("-gt", "--gpuType", required=True, help = """pascal|turing|hiai|hisi""")
ap.add_argument("-gd", "--gpuId", required=False, default="1", help = "gpu id")
ap.add_argument("-zp", "--zips", required=False, default="", help = """vehicleDetect.zip,plateChar.zip,plateColor.zip,plateRec.zip""")
ap.add_argument("-it", "--inputs", required=True, default="", help = "jpg|jpgDirectory|jpgList")
ap.add_argument("-vt", "--virtualize", required=False, default="0", help = """0:disable 1:enable""")

args = vars(ap.parse_args())
gpuType = args.get("gpuType")
gpuId = args.get("gpuId")
zips = args.get("zips")
inputs = args.get("inputs")
virtualize = args.get("virtualize")

# get all files under directory
def getFilesUnderDir(top, filepat="*.*"):
    '''
    Find all filenames in a directory tree that match a shell wildcard pattern
    '''
    for path, dirlist, filelist in os.walk(top):
        for name in fnmatch.filter(filelist, filepat):
            if name.endswith((".jpg", ".JPG", ".png", ".PNG", ".bmp", ".BMP")):
                yield os.path.join(path,name)

# deal with image/directory/list
def dealInput(inputs):

    cwd = os.getcwd()
    vd = open('vehicleDetect.list', 'w')

    if os.path.isdir(inputs):
        allLines = getFilesUnderDir(inputs)
        for line in allLines:
            if line.startswith("/"):
                pass
            else:
               line = cwd + "/" + line
            vd.write('"{}"'.format(line))
            vd.write(os.linesep)
    else:
        if 'text' in magic.from_file(inputs, mime=True): 
            allLines = open(inputs).readlines()
            for line in allLines:
                vd.write('"{}"'.format(line))
        elif 'image' in magic.from_file(inputs, mime=True):
            if inputs.startswith("/"):
                allLines = inputs
            else:
                allLines = cwd + "/" + inputs
            vd.write('"{}"'.format(allLines))

    vd.close()

# manual set gpu id
def manualGpu(session=None, gpuType=None, gpuId=None):

    if gpuType == 'pascal' or gpuType == 'turing': # cuda
        session.sendline("""sed  -i -e 's/.*Device.*/    "Device": {},/g' ci/environment/trt_60_{}.json""".format(gpuId, gpuType))
        session.expect("#", timeout=None)
    elif gpuType == 'hisi': # nnie (hisi)
        pass
    elif gpuType == 'hiai': # atlas (hiai)
        session.sendline("""sed  -i -e 's/.*Device.*/    "Device": {},/g' ci/environment/hiai_b896.json""".format(gpuId))
        session.expect("#", timeout=None)
    elif gpuType == 'm40': # m40
        session.sendline("""sed  -i -e 's/.*Device.*/    "Device": {},/g' ci/environment/trt_60_m40_148.json""".format(gpuId))
        session.expect("#", timeout=None)

    print('manualGpu set ok')

# modify the vehicle detect json
def modifyVehicleDetectJson():

    cwd = os.getcwd()

    inputFile = cwd + "/" + "vehicleDetect.list"
    data = {"precision":[{"Input":inputFile,"Output":"result.VehicleDetector"}]}
    with open('ci/case/VehicleDetector.json', 'w') as f:
        json.dump(data, f)

    #mvd = open("ci/case/VehicleDetector.json", "w")
    #mvd.write('''{"precision":[{"Input":{},"Output":"result.VehicleDetector"}]}'''.format(inputFile))
    #mvd.close()

# get the docker session
def getDockerSession(prompt='#'):

    session = pexpect.spawn("/bin/bash", timeout=None, encoding='utf-8')

    fout = open('ssh.txt','w')
    session.logfile = fout
    session.expect(prompt)
    #session.read()
    return session

# run the case
def runCase(session=None, gpuType=None, modelName=None, prompt='#'):

    cwd = os.getcwd()

    if gpuType == 'pascal' or gpuType == 'turing': # cuda
        jsonFile = 'ci/environment/trt_60_{}.json'.format(gpuType)
        modelFile = 'ci/model_list/trt_60_{}_all.txt'.format(gpuType)
        releasePath = "release/cuda/"
    elif gpuType == 'hisi': # nnie
        jsonFile = ""
        modelFile = 'ci/model_list/nnie.txt'
        releasePath = ""
    elif gpuType == 'hiai': # atals
        jsonFile = 'ci/environment/hiai_b896.json'
        modelFile = 'ci/model_list/hiai_b896_all.txt'
        releasePath = "release/hiai/"
    elif gpuType == 'm40': # m40
        jsonFile = 'ci/environment/trt_60_m40_148.json'
        modelFile = 'ci/model_list/trt_60_maxwell_all.txt'
        releasePath = "release/cuda/"

    if modelName != 'PlateGraph':
        cmds = """
        bash {cwd}/ci/run_case.sh -e {jsonFile} -m `grep -i {modelName} {modelFile}` -r {releasePath} -t precision -o {cwd}
        """.format(jsonFile=jsonFile, modelName=modelName, modelFile=modelFile, releasePath=releasePath, cwd=cwd)
    else:
        '''
        for item in ['PlateChar', 'PlateColor', 'PlateRectify']:
            session.sendline("grep {} {} >> ci/model_list/plate_graph.list".format(item, modelFile))
            session.expect("ci")

        session.sendline("echo ci/model_group/PlateGraph.json >> ci/model_list/plate_graph.list")
        session.expect("ci")
        '''

        cmds = """
        bash {cwd}/ci/run_case.sh -e {jsonFile} -m ci/model_list/plate_graph.list -r {releasePath} -t precision -o {cwd}
        """.format(jsonFile=jsonFile, modelName=modelName, modelFile=modelFile, releasePath=releasePath, cwd=cwd)

    cmds = [cmds.strip("\r\n ")]
    print('cmds:', cmds)
    for cmd in cmds:
        session.sendline(cmd)
        if 'run_case.sh' in cmd:
            if 'plate_graph' not in cmd:
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
                    sys.exit(1)
            else:
                match = 'PlateRectify'
                expectMatch = "{} test is finished".format(match)
                ret = session.expect([expectMatch, 'Check failed: error', 'error code is'], timeout=None)
                print('ret:', ret)
                if ret == 0:
                    pass
                elif ret == 1 or ret == 2:
                    buffer = session.before[len(session.before)-2000:]
                    sys.exit(1)
        else:
            session.expect(cmd[:5], timeout=None)

        session.expect(prompt, timeout=None)
        session.buffer = ""

    session.sendline('ls')
    session.expect('ls')

# crop the vehicle
def vehicleCrop(rtName):

    if os.path.exists("plateCrop"):
        os.system("rm -rf plateCrop")
        os.system("mkdir plateCrop")
    else:
        os.system("mkdir plateCrop")

    rtDict = {}
    with open(rtName) as rt:
        for line in rt:
            line = line.strip("\r\n")
            match = re.search("(.*jpg|.*.JPG|.*png|.*bmp)(.*)", line)
            if match:
                img = match.group(1)
                lines = match.group(2)

            '''
            if '.jpg ' in line:
                lines = line.split(".jpg ")
            else:
                lines = line.split(".jpg")
            img = lines[0] + '.jpg'
            '''

            try:
                linesLeft = lines.strip().split(" ")
                if img not in rtDict:
                    rtDict[img] = []
            except:
                rtDict[img] = []
                continue
            numby6 = len(linesLeft)//6
            for i in range(numby6):
                startIndex = i*6
                endIndex = (i+1)*6
                items = linesLeft[startIndex:endIndex]
                c,t,x,y,w,h = items
                rtDict[img].append([x,y,w,h,t])

    for img in rtDict:

        #print('img:', img)
        rRois = rtDict[img]
        im = cv2.imread(img)
        im_copy = im.copy()

        for oRoi in rRois:
            x, y, w, h, t = oRoi
    
            if int(t) != 1:
                continue
            x1 = int(x)
            y1 = int(y)
            x2 = int(x) + int(w)
            y2 = int(y) + int(h)
            cropped = im_copy[y1:y2, x1:x2]
            cv2.imwrite("plateCrop/{}_{}_{}_{}_{}".format(x1, y1, x2, y2, os.path.basename(img)), cropped)

# make list for plate detect
def makeListPlateDetect(inputs="plateCrop"):

    cwd = os.getcwd()

    pd = open("plateDetect.list", "w")
    allLines = getFilesUnderDir(cwd + "/" + inputs) 
    for line in allLines:
        pd.write('"{}"'.format(line))
        pd.write(os.linesep)
    pd.close() 

# modify the plate detect json
def modifyPlateDetectJson():

    cwd = os.getcwd()

    inputFile = cwd + "/" + "plateDetect.list"
    data = {"precision":[{"Input":inputFile,"Output":"result.PlateDetector"}]}
    with open('ci/case/PlateDetector.json', 'w') as f:
        json.dump(data, f)

# build list of plategraph
def buildListPlateGraph():
    pg = open('plategraph.list', 'w')
    with open('result.PlateDetector' + getResultTail()) as pd:
        for line in pd:
            lines = line.strip("\r\n").split(" ")
            if len(lines) == 1:
                continue
            else:
                img, c, t, x, y, w, h = lines
                pg.write('"{}",{},{},{},{}{}'.format(img,x,y,w,h,os.linesep))
    pg.close()

# modify the plate char/color/recity json
def modifyPlateJson():

    cwd = os.getcwd()

    inputFile = cwd + "/" + "plategraph.list"
    data = {"precision":[{"Input":inputFile,"Output":"result.PlateGraph"}]}
    with open('ci/case/PlateGraph.json', 'w') as f:
        json.dump(data, f)

    inputFile = "/mnt/cephfs/testData/plate/plate_all_color/" + "plate_char_oneline.list"
    data = {"precision":[{"Input":inputFile,"Output":"result.PlateChar"}]}
    with open('ci/case/PlateChar.json', 'w') as f:
        json.dump(data, f)

    inputFile = "/mnt/cephfs/testData/plate/plate_all_color/" + "plate_color_oneline.list"
    data = {"precision":[{"Input":inputFile,"Output":"result.PlateColor"}]}
    with open('ci/case/PlateColor.json', 'w') as f:
        json.dump(data, f)

    inputFile = "/mnt/cephfs/testData/plate/plate_all_color/" + "plate_rectify_oneline.list"
    data = {"precision":[{"Input":inputFile,"Output":"result.PlateRectify"}]}
    with open('ci/case/PlateRectify.json', 'w') as f:
        json.dump(data, f)
   
# build the plategraph
def buildPlateGraphZip():

    if gpuType == 'pascal' or gpuType == 'turing': # cuda
        jsonFile = 'ci/environment/trt_60_{}.json'.format(gpuType)
        modelFile = 'ci/model_list/trt_60_{}_all.txt'.format(gpuType)
        releasePath = "release/cuda/"
    elif gpuType == 'hisi': # nnie
        jsonFile = ""
        modelFile = 'ci/model_list/nnie.txt'
        releasePath = ""
    elif gpuType == 'hiai': # atals
        jsonFile = 'ci/environment/hiai_b896.json'
        modelFile = 'ci/model_list/hiai_b896_all.txt'
        releasePath = "release/hiai/"
    elif gpuType == 'm40': # m40
        jsonFile = 'ci/environment/trt_60_m40_148.json'
        modelFile = 'ci/model_list/trt_60_maxwell_all.txt'
        releasePath = "release/cuda/"

    print('zips:', zips)
    if zips == '':
        session = getDockerSession()
        session.sendline("> ci/model_list/plate_graph.list")
        session.expect("ci")
        for item in ['PlateChar', 'PlateColor', 'PlateRectify']:
            session.sendline("grep {} {} >> ci/model_list/plate_graph.list".format(item, modelFile))
            session.expect("ci")
        session.sendline("echo ci/model_group/PlateGraph.json >> ci/model_list/plate_graph.list")
        session.expect("ci")
        session.sendline("cat ci/model_list/plate_graph.list")
        session.expect("ci")
    else:
        vehicleDetect,plateChar,plateColor,plateRectify = zips.split(",")
        with open('ci/model_list/plate_graph.list', 'w') as cw:
            cw.write("{}{}".format(plateChar,os.linesep))
            cw.write("{}{}".format(plateColor,os.linesep))
            cw.write("{}{}".format(plateRectify,os.linesep))
            cw.write("ci/model_group/PlateGraph.json")

# get the result tail
def getResultTail():

    if gpuType == 'turing':
        tail = '_trt_60_turing'
    elif gpuType == 'pascal':
        tail = '_trt_60_pascal'
    elif gpuType == 'hiai':
        tail = '_hiai_b896'

    return tail

# virtualize 
def plateVirtualize():

    if os.path.exists('plateVir'):
        os.system("rm -rf plateVir")
        os.system("mkdir plateVir")
    else:
        os.system("mkdir plateVir")

    with open('result.PlateGraph' + getResultTail()) as pt:
        for line in pt:
            match = re.search(r"(.*?); color: (\d+),.*?character: (.*?);", line)
            if match:
                img = match.group(1)
                color = colorDict[match.group(2)]
                char = match.group(3)

                im = cv2.imread(img)
                im_copy = im.copy()
                img_pil = Image.fromarray(im_copy)
                draw = ImageDraw.Draw(img_pil)
                b, g, r, a = 255, 0, 0, 0
                draw.text((20,20), color.strip("\r\n"), font = pfont, fill = (b, g, r, a))
                draw.text((20,40), char.strip("\r\n"), font = pfont, fill = (b, g, r, a))
                im_copy = np.array(img_pil)
                cv2.imwrite("plateVir/" + os.path.basename(img) , im_copy)

if __name__ == '__main__':
    dealInput(inputs) # deal with image/directory/list	
    modifyVehicleDetectJson() # modify the vehicle detect json	
    docSess = getDockerSession()
    manualGpu(docSess, gpuType, gpuId)
    runCase(docSess, gpuType, modelName="VehicleDetector_")
    vehicleCrop('result.VehicleDetector' + getResultTail())
    makeListPlateDetect()
    modifyPlateDetectJson()
    runCase(docSess, gpuType, modelName="PlateDetector")
    buildListPlateGraph()
    modifyPlateJson()
    buildPlateGraphZip()
    runCase(docSess, gpuType, modelName="PlateGraph")
    if int(virtualize) == 1:
        plateVirtualize() 
