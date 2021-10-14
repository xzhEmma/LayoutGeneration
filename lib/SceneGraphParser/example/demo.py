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

#import SceneGraphParser.sng_parser
import sng_parser
from collections import defaultdict
import json
import copy
from nltk.corpus import wordnet as wn


def txt_to_matrix(filename):
    file = open(filename)
    lines = file.readlines()
    # print lines
    # ['0.94\t0.81\t...0.62\t\n', ... ,'0.92\t0.86\t...0.62\t\n']形式
    rows = len(lines)  # 文件行数

    datamat = np.zeros((rows, 3))  # 初始化矩阵

    row = 0
    for line in lines:
        line = line.strip().split('\t')  # strip()默认移除字符串首尾空格或换行符
        datamat[row, :] = line[:]
        row += 1

    return datamat


def scene_graph_dict(file_path):
    a=txt_to_matrix(file_path)
    graphs=[]
    for i in range(a.shape[1]):
        graph=sng_parser.parse(sng_parser.parse(sentences), show_entities=False)
        graphs.append(graph)
    print(graphs)

def main():
    scene_graph_dict("../../ data / AbstractScenes_v1.1 / SimpleSentences / SimpleSentences1_10020.txt")

if __name__ == '__main__':
    main()
