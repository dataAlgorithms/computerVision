import json
import sys
import os

fname = sys.argv[1]

dataDict = {}
with open('new.' + fname, 'w') as fw, open('detect.' + fname, 'w') as dw:
    with open(fname) as f:
        lines = (line.strip() for line in f)
        for line in lines:
            data = json.loads(line)
            img = data['url_image']
            results = data['result']
            fw.write("{}".format(img))
            #if len(results) != 25:
            #    print(line)
            #    input()
            if len(results) != 25:
                continue

            dw.write("{}".format(os.path.basename(img)))
            for result in results:
                data = result['data']
                tagtype = result['tagtype']
                if tagtype == 'body':
                    x, y, w, h = data
                    w = int(x) + int(w)
                    h = int(y) + int(h)
                    dw.write(" 0.99 {} {} {} {}{}".format(x,y,w,h,os.linesep))
                    continue
                for item in data:
                    fw.write(",{}".format(item))
            fw.write(os.linesep)
