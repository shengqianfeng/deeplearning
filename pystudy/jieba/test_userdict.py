#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
@File : test_userdict.py.py
@Author : jeffsheng
@Date : 2019/11/21
@Desc : 自定义分词
①开发者可以指定自己自定义的词典，以便包含 jieba 词库里没有的词。虽然 jieba 有新词识别能力，但是自行添加新词可以保证更高的正确率
②一个词占一行；每一行分三部分：词语、词频（可省略）、词性（可省略），用空格隔开，顺序不可颠倒
③词频省略时使用自动计算的能保证分出该词的词频。
"""

from __future__ import print_function, unicode_literals
import sys
sys.path.append("../")
import jieba
jieba.load_userdict("userdict.txt")
import jieba.posseg as pseg

jieba.add_word('石墨烯')
# jieba.add_word('凱特琳')
# jieba.del_word('自定义词')

test_sent = (
"李小福是创新办主任也是云计算方面的专家; 什么是八一双鹿\n"
"例如我输入一个带“韩玉赏鉴”的标题，在自定义词库中也增加了此词为N类\n"
"「台中」正確應該不會被切開。mac上可分出「石墨烯」；此時又可以分出來凱特琳了。"
)
words = jieba.cut(test_sent)
print('/'.join(words))

print("="*40)

result = pseg.cut(test_sent)
for w in result:
    print(w.word, "/", w.flag, ", ", end=' ')
print("\n" + "="*40)

terms = jieba.cut('easy_install is great')
print('/'.join(terms))  # easy_install/ /is/ /great
terms = jieba.cut('python 的正则表达式是好用的')
print('/'.join(terms))  # python/ /的/正则表达式/是/好用/的

print("="*40)

# test frequency tune 频率调谐
testlist = [
('今天天气不错', ('今天', '天气')),
('如果放到post中将出错。', ('中', '将')),
('我们中出了一个叛徒', ('中', '出')),
]

for sent, seg in testlist:
    print('/'.join(jieba.cut(sent, HMM=False)))
    word = ''.join(seg)
    print('%s Before: %s, After: %s' % (word, jieba.get_FREQ(word), jieba.suggest_freq(seg, True)))
    print('/'.join(jieba.cut(sent, HMM=False)))
    print("-"*40)