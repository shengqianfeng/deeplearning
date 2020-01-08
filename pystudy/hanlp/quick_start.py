#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
@File : quickstart.py
@Author : jeffsheng
@Date : 2020/1/8
@Desc : hanlp的python用法
"""

from pyhanlp import *

print(HanLP.segment('我想听刘德华的忘情水'))
# 获取单词与词性
for term in HanLP.segment('我想听刘德华的忘情水'):
    print('{}\t{}'.format(term.word, term.nature))

testCases = [
    "我想听刘德华的忘情水",
    "来一首歌"]
for sentence in testCases:
    print(HanLP.segment(sentence))

print("----------抽取关键词------------")
document = "我想听刘德华的忘情水，" \
           "来一首歌"
print(HanLP.extractKeyword(document, 2))
print("=----------自动摘要--------")
print(HanLP.extractSummary(document, 3))

print("-------------依存句法分析-----------")
print(HanLP.parseDependency("我想听刘德华的忘情水、来一首歌。"))
print("-------------------------------------------")
print("-------------------------------------------")
print("------------------加载确属词和意图词，确定句子所属领域-------------------------")