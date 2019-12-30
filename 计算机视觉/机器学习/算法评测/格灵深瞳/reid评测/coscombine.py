import numpy as np
import sys
import numpy
import base64
from optparse import OptionParser
from itertools import combinations
import numpy as np
import sys
import numpy
import base64
from optparse import OptionParser
import os

allFeas = [line.strip().split(" ")[0] + ':' + line.strip().split(" ")[1] for line in open(sys.argv[1]).readlines()]
nObj =  open('fea.score',  'w')

def feature2array(feature):
    dec = base64.decodestring(feature)
    array = numpy.frombuffer(dec,dtype=numpy.float32)
    s = base64.b64encode(array)

    sumsquare = 0
    for num in array:
        sumsquare += num * num

    return array

for i in combinations(allFeas, 2):
    vector1, vector2 = i

    img1, vector1 = vector1.split(":")
    img2, vector2 = vector2.split(":")

    vector1 = np.array(feature2array(vector1))
    vector2 = np.array(feature2array(vector2))
    op7=np.dot(vector1,vector2)/(np.linalg.norm(vector1)*(np.linalg.norm(vector2)))
    score = (op7+1)/2.0

    if score > 1:
        score = 1
    if score < 0:
        score = 0

    nObj.write('{} {} score:{}{}'.format(img1, img2, score, os.linesep))
nObj.close()
