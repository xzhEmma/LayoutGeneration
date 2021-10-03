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


def demo(sentence):
    print('Sentence:', sentence)  # 输出句子
    relation_data = sng_parser.tprint(sng_parser.parse(sentence), show_entities=False)
    # t print函数是把场景图的信息以表格形式打印下来，parse是一个对文本进行解析的函数
    return relation_data


def main_val():  # 我理解这个函数就是将这个json文件的关系出现次数都记录下来
    caption_rel_lst = []
    annotations_file_path = "captions_train-val2017/annotations/captions_val2017.json"
    coco = COCO(annotations_file_path)
    caption_dict = coco.anns
    for (k, v) in caption_dict.items():
        caption = v["caption"]
        relation_data = demo(caption)

        for i in range(len(relation_data)):
            triple = relation_data[i]
            triple_rel = triple[1]
            print("rel", triple_rel)
            # caption_rel_set.add(triple_rel)
            caption_rel_lst.append(triple_rel)

    rel_coco = Counter(caption_rel_lst)

    print("caption_set", rel_coco)
    b = json.dumps(rel_coco)
    f2 = open('data_set/caption_val_rel.json', 'w')
    f2.write(b)
    f2.close()



def main_train():
    caption_rel_lst = []
    annotations_file_path = "captions_train-val2017/annotations/captions_train2017.json"
    coco = COCO(annotations_file_path)
    caption_dict = coco.anns
    for (k, v) in caption_dict.items():
        caption = v["caption"]
        relation_data = demo(caption)

        for i in range(len(relation_data)):
            triple = relation_data[i]
            triple_rel = triple[1]
            print("rel", triple_rel)
            # caption_rel_set.add(triple_rel)
            caption_rel_lst.append(triple_rel)

    rel_coco = Counter(caption_rel_lst)

    b = json.dumps(rel_coco)
    f2 = open('data_set/caption_train_rel.json', 'w')
    f2.write(b)
    f2.close()


def translation(content):

    url = 'http://fanyi.youdao.com/translate?smartresult=dict&smartresult=rule'
    data = {'i': content, 'doctype': 'json'}  # 定义一个字典，将data数据存入

    data = urllib.parse.urlencode(data).encode('utf-8')
    response = urllib.request.urlopen(url, data)
    html = response.read().decode('utf-8')
    target = json.loads(html)
    print("翻译的结果为 : %s" % (target['translateResult'][0][0]['tgt']))


def main_all():
    with open('data_set/caption_train_rel.json', 'r', encoding='utf8')as fptrain:
        train_rel_data = json.load(fptrain)
    with open('data_set/caption_val_rel.json', 'r', encoding='utf8')as fpval:
        val_rel_data = json.load(fpval)
    with open('Geometric.json', 'r', encoding='utf8')as g:
        G = json.load(g)
    with open('data_set/Possessive.json', 'r', encoding='utf8')as p:
        P = json.load(p)
    with open('data_set/Semantic.json', 'r', encoding='utf8')as s:
        S = json.load(s)
    with open('data_set/Misc.json', 'r', encoding='utf8')as m:
        M = json.load(m)

    A = Counter(train_rel_data)   # 计数函数
    B = Counter(val_rel_data)
    C = A + B

    geometric = {}
    possessive = {}
    semantic = {}
    misc = {}
    i = 0

    for key, value in C.items():
        i = i + 1
        print("这是第%d次" % i)
        print(key, value)
        if key in G:
            continue
        elif key in P:
            continue
        elif key in S:
            continue
        elif key in M:
            continue
        elif value >= 50:
            translation(key)
            relation = int(input("请判断它是属于哪种关系:\n1代表Geometric，2代表Possessive，3代表Semantic，4代表Misc\n "))
            if relation == 1:
                geometric[key] = value
                print("hh")
            if relation == 2:
                possessive[key] = value
            if relation == 3:
                semantic[key] = value
            if relation == 4:
                misc[key] = value
            D = json.dumps(geometric)
            E = json.dumps(possessive)
            F = json.dumps(semantic)
            G = json.dumps(misc)

            f1 = open('Geometric1.json', 'w')
            f1.write(D)
            f2 = open('Possessive1.json', 'w')
            f2.write(E)
            f3 = open('Semantic1.json', 'w')
            f3.write(F)
            f4 = open('Misc1.json', 'w')
            f4.write(G)
            f1.close()
            f2.close()
            f3.close()
            f4.close()



if __name__ == '__main__':
    # main_val()
    # main_train()
    main_all()
