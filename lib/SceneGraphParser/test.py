from collections import defaultdict
import sng_parser
import os
import nltk
from pycocotools.coco import COCO
import numpy as np
import matplotlib.pyplot as plt
import urllib.request
import urllib.parse
import json
import pylab

pylab.rcParams['figure.figsize'] = (8.0, 10.0)
import json

from pycocotools.coco import COCO
from collections import Counter


def translation(content):

    url = 'http://fanyi.youdao.com/translate?smartresult=dict&smartresult=rule'
    data = {'i': content, 'doctype': 'json'}  # 定义一个字典，将data数据存入

    data = urllib.parse.urlencode(data).encode('utf-8')
    response = urllib.request.urlopen(url, data)
    html = response.read().decode('utf-8')
    target = json.loads(html)
    print("翻译的结果为 : %s" % (target['translateResult'][0][0]['tgt']))


def main_all():
    with open('data_set/caption_all_rel.json', 'r', encoding='utf8')as train:
        C = json.load(train)

    s = defaultdict(list)

    i = 0

    for key in C.keys():
        i = i + 1
        print("这是第%d次" % i)
        print(key)
        translation(key)
        j = 1
        print("请判断它属于哪种关系\n1代表above，2代表below，3代表right of，4代表left of，5代表surrounding，6代表inside,0代表判断结束")
        while j != 0:
            j = int(input())
            if j == 1:
                s[key].append('above')
            elif j == 2:
                s[key].append('below')
            elif j == 3:
                s[key].append('right of')
            elif j == 4:
                s[key].append('left of')
            elif j == 5:
                s[key].append('surrounding')
            elif j == 6:
                s[key].append('inside')
        R = json.dumps(s, indent=4)
        f1 = open('data_set/all.json', 'w+')
        f1.write(R)
        f1.close()



if __name__ == '__main__':
    # main_val()
    # main_train()
    main_all()



