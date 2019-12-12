import json
import sys
import os

fname = sys.argv[1]

dataDict = {}
with open('new.' + fname, 'w') as fw:
    with open(fname) as f:
        lines = (line.strip() for line in f)
        for line in lines:
            data = json.loads(line)
            img = data['url_image']
            results = data['result']
            fw.write("{}".format(img))
            for result in results:
                x,y,w,h = result['data']
                fw.write(",2,{},{},{},{}".format(x,y,w,h))
            fw.write(os.linesep)
