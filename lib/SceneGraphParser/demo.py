#! /usr/bin/env python3
# -*- coding: utf-8 -*-
# File   : demo.py
# Author : Jiayuan Mao
# Email  : maojiayuan@gmail.com
# Date   : 08/22/2018
#
# This file is part of SceneGraphParser.
# Distributed under terms of the MIT license.
# https://github.com/vacancy/SceneGraphParser

"""
A small demo for the scene graph parser.
"""

import sng_parser
from collections import defaultdict
import json
import copy
from nltk.corpus import wordnet as wn
import numpy as np
import json
import os



def txt_strtonum_feed(filename):
    sentences=[]
    with open(filename, 'r') as f:
        line = f.readline()
        while line:
            eachline = line.replace('\n','').split('\t')###按行读取文本文件，每行数据以列表形式返回
            sentences.append(eachline)
            line = f.readline()
            line = f.readline()
        return sentences

def save_txt_results(text, count, text_dir):
    if not os.path.exists(text_dir):
                os.makedirs(text_dir)
    f = open('%s/graph_%05d.txt'%(text_dir, count), 'a')
    f.write(json.dumps(text))
    f.close()

def scene_graph_dict(file_path,out_dir,n=10020):
    sentences=txt_strtonum_feed(file_path)
    #print(len(sentences))
    num_sen=len(sentences)
    sen=[]
    #句子分组合并，大多数是三句.
    for j in range(n):
        for i in range(num_sen):
            if int(sentences[i][0])==j:
                if int(sentences[i][1])==0:
                    sen.append(sentences[i][2])
                else:
                    #print(sentences[i])
                    sen[j]=sen[j]+sentences[i][2]
    #将列表写入txt文件
    f=open('./Sentences','w+')
    for line in sen:
        f.write(line + '\n')
    f.close()
    j=0
    with open('./Sentences') as f:
        line = f.readline()
        while line:
           # print(j)
            graph=sng_parser.parse(line) 
            save_txt_results(graph,j ,out_dir)
            
            j=j+1
            line = f.readline()


def build_graph(path):
    with open(path, 'rb') as f:
        desc = f.read()
        desc = eval(desc)
        desc_entities = desc['entities']
        desc_relations = desc['relations']
        # count the number of nodes
        count_nodes = len(desc_entities)
        adj = np.zeros((count_nodes, count_nodes))
        # handle adjacent
        for desc_relation in desc_relations:
            sub = desc_relation['subject']
            obj = desc_relation['object']
            # rel = desc_relation['relation']
            adj[sub][obj] = 1
   # print(adj)
    return adj

# def obj_extra(obj)
#
#     input_inds, input_lens = encode_sentences(obj)
#     out = TextEncoder(input_inds, input_lens)
#     return out

def main():
    scene_graph_dict('/root/Text2Scene/data/AbstractScenes_v1.1/SimpleSentences/SimpleSentences2_10020.txt','/root/Text2Scene/data/graphs_txt2')
    #build_graph('/tmp/text2scene/data/AbstractScenes_v1.1/graph_txt/count_000001.txt')

if __name__ == '__main__':
    main()
