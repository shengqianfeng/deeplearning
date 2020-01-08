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
    for word in sent:
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

