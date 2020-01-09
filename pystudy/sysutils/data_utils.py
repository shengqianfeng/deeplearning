#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
@File : data_utils.py
@Author : jeffsheng
@Date : 2020/1/8
@Desc : 
"""
import pickle
import os
import re


def read_dictionary(vocab_path):
    """
    读取储存的字典，pkl文件
    :param vocab_path:
    :return:
    """
    vocab_path = os.path.join(vocab_path)
    with open(vocab_path, 'rb') as fr:
        word2id = pickle.load(fr)
    print('vocab_size:', len(word2id))
    return word2id



def get_segmented_vector(line, length, word2idx):
    tokenized_line = sentence2id(line, word2idx)
    if len(tokenized_line) > length:
        del tokenized_line[length:]
    else:
        k = length - len(tokenized_line)
        tokenized_line += [word2idx["<UNK>"]] * k

    if len(tokenized_line) != length:
        print(tokenized_line)

    return [tokenized_line]


def sentence2id(sent, word2idx):
    """
    句子转数字索引
    :param sent:
    :param word2idx:
    :return:
    """
    sentence_id = []
    for word in sent.split():
        if word.isdigit():
            word = "<NUM>"
        elif word.encode('UTF-8').isalpha():
            word = word.lower()
        elif word.isspace(): # 为空字符
            word = "#"
        if word not in word2idx:
            word = "<UNK>"
        sentence_id.append(word2idx[word])
    return sentence_id
# import tensorflow as tf

def clean_text(text_string):
    text_string = re.sub(r'([^\s\w]|_|[0-9])+', '', text_string)
    text_string = " ".join(text_string.split())
    text_string = text_string.lower()
    return text_string


#
#
#
# import numpy as np
# # 调用 clean_text 清理文本
# s='Go until jurong point, crazy.. Available only in bugis n great world la e buffet... Cine there got amore wat...'
# text = clean_text(s)
#
# word2idx = read_dictionary('../deep_learning/gitchat/vocab/word2id.pkl')
# x = sentence2id(text,word2idx._mapping)
# print(x)