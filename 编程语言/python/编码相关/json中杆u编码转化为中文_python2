#!/usr/bin/env python
#!coding=utf-8

import json
import os
import sys
reload(sys)
sys.setdefaultencoding('utf8')

filename = sys.argv[1]
with open('new.' + filename, 'w') as fn:
    with open(filename) as fi:
        for idx, line in enumerate(fi):
            i_json = json.loads(line)
            fn.write(json.dumps(i_json, ensure_ascii=False) + '\n')
