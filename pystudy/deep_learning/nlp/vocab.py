#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
@File : vocab.py
@Author : jeffsheng
@Date : 2020/1/7
@Desc : tensorflow词向量处理
"""


import tensorflow as tf
import numpy as np


print(tf.__version__)
max_document_length = 4
x_text =[
    'i love you',
    'me too'
]
"""
tf.contrib.learn.preprocessing.VocabularyProcessor (max_document_length, min_frequency=0, vocabulary=None, tokenizer_fn=None)
- max_document_length: 是文本的最大长度。如果文本的长度大于这个值，就会被剪切，小于这个值的地方用 0 填充。
- min_frequency: 是词频的最小值。当单词的出现次数小于这个词频，就不会被收录到词表中。
- vocabulary: CategoricalVocabulary 对象。
- tokenizer_fn：分词函数

使用这个函数时一般分为几个动作：
1.首先将列表里面的词生成一个词典；
2.按列表中的顺序给每一个词进行排序，每一个词都对应一个序号(从1开始，<UNK>的序号为0)
3.按照原始列表顺序，将原来的词全部替换为它所对应的序号
4.同时如果大于最大长度的词将进行剪切，小于最大长度的词将进行填充
5.然后将其转换为列表，进而转换为一个array
"""
vocab_processor = tf.contrib.learn.preprocessing.VocabularyProcessor(max_document_length)
x = np.array(list(vocab_processor.fit_transform(x_text)))
'''
[[1 2 3 0]
 [4 5 0 0]]
'''
print(x)
print(len(vocab_processor.vocabulary_))
vocab_dict = vocab_processor.vocabulary_._mapping
# {'<UNK>': 0, 'i': 1, 'love': 2, 'you': 3, 'me': 4, 'too': 5}
print(vocab_dict)

sorted_vocab = sorted(vocab_dict.items(), key=lambda x: x[1])
# [('<UNK>', 0), ('i', 1), ('love', 2), ('you', 3), ('me', 4), ('too', 5)]
print(sorted_vocab)

vocabulary = list(list(zip(*sorted_vocab))[0])
# ['<UNK>', 'i', 'love', 'you', 'me', 'too']
print(vocabulary)

w=list(zip(*sorted_vocab))
# [('<UNK>', 'i', 'love', 'you', 'me', 'too'), (0, 1, 2, 3, 4, 5)]
print(w)

print("---------embedding_lookup函数的使用------")
# 生成10*1的张量
p = tf.Variable(tf.random_normal([10,1]))
# 查找张量中的序号为1和3的
b = tf.nn.embedding_lookup(p, [1, 3])
with tf.Session() as sess:
    sess.run(tf.global_variables_initializer())
    print(sess.run(p))
    print(sess.run(b))
    print(p)
    print(type(p))
