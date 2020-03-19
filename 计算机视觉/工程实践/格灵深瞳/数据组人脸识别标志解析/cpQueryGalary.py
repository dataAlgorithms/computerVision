import sys
import re
import os
import fnmatch

labelName = sys.argv[1]
fObj = open('map.txt', 'w')

def buildGt():
    aDict = {}
    with open(labelName) as ln:
        for line in ln:
            line = line.strip("\r\n")
            img, gid, ctype = line.split(" ")
            if gid not in aDict:
                aDict[gid] = [ctype + ':' + img + ':' + gid]
            else:
                aDict[gid].append(ctype + ':' + img + ':' + gid)

            aDict[gid] = sorted(aDict[gid], key=lambda item: int(item.split(":")[0]))
    return aDict

def gen_find(filepat="*", top=None):
    '''
    Find all filenames in a directory tree that match a shell wildcard pattern
    '''
    for path, dirlist, filelist in os.walk(top):
        for name in fnmatch.filter(filelist, filepat):
            yield os.path.join(path,name)

def indexCheck(img, gt):

    #print('img:',  img)
    
    flag = 0
    for key in gt:
        items = gt[key]
        for index, item in enumerate(items):
            if item.startswith('1:{}'.format(img)):
                galary = items[0]
                flag = 1
                break
        if flag == 1:
            break

    if flag == 1:
        
        #print('query:', img)
        #print('galary:', galary)
        galarySp = galary.split(":")[1]
        gid = galary.split(":")[-1]

        #print('galary:{} gid:{}'.format(galary, gid))
        #raw_input()
        os.system("cp {} queryCandidate/{}_{}.jpg".format(query, gid, index))
        fObj.write("cp {} queryCandidate/{}_{}.jpg{}".format(query, gid, index, os.linesep))
        os.system("cp {} galaryCandidate/{}_0.jpg".format(galarySp, gid)) 
        fObj.write("cp {} galaryCandidate/{}_0.jpg{}".format(galarySp, gid, os.linesep)) 
 
gt = buildGt()

querys = gen_find(filepat="*.jpg", top="extrace_frames")
for index, query in enumerate(querys):
    indexCheck(query, gt)

fObj.close()
