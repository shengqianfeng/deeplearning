"""
word2vec的实现
    我们将在PTB训练集上训练词嵌入模型。该数据集的每一行作为一个句子。句子中的每个词由空格隔开

PTB（Penn Tree Bank）是一个常用的小型语料库
它采样自《华尔街日报》的文章，包括训练集、验证集和测试集
"""


import collections
import d2lzh as d2l
import math
from mxnet import autograd, gluon, nd
from mxnet.gluon import data as gdata, loss as gloss, nn
import random
import sys
import time
import zipfile

print("------处理数据集--------------")
with zipfile.ZipFile('../data/ptb.zip', 'r') as zin:
    zin.extractall('../data/')

with open('../data/ptb/ptb.train.txt', 'r') as f:
    lines = f.readlines()
    # st是sentence的缩写
    raw_dataset = [st.split() for st in lines]

print('# sentences: %d' % len(raw_dataset))

# 对于数据集的前3个句子，打印每个句子的词数和前5个词。
# 这个数据集中句尾符为“<eos>”，生僻词全用“<unk>”表示，数字则被替换成了“N”。
for st in raw_dataset[:3]:
    print('# tokens:', len(st), st[:5])

"""
打印如下：
# tokens: 24 ['aer', 'banknote', 'berlitz', 'calloway', 'centrust']
# tokens: 15 ['pierre', '<unk>', 'N', 'years', 'old']
# tokens: 11 ['mr.', '<unk>', 'is', 'chairman', 'of']
"""
print("-------------建立词语索引-----------------")
# 我们只保留在数据集中至少出现5次的词
# tk是token的缩写 从数据集raw_dataset中每次拿出一行st出来，迭代st的每个单词tk
counter = collections.Counter([tk for st in raw_dataset for tk in st])# 计算每个单词出现的次数
counter = dict(filter(lambda x: x[1] >= 5, counter.items()))# 获取次数出现大于5的单词列表
# 然后将词映射到整数索引
idx_to_token = [tk for tk, _ in counter.items()]# {索引：词}
token_to_idx = {tk: idx for idx, tk in enumerate(idx_to_token)}# {词：索引}
dataset = [[token_to_idx[tk] for tk in st if tk in token_to_idx]
           for st in raw_dataset]# 给数据集转化为索引数据集，单词全部转化为索引
num_tokens = sum([len(st) for st in dataset])# 统计总单词数
print('# tokens: %d' % num_tokens)  # # tokens: 887100

print("-----------------二次采样------------------------------------------")
#   f(wi)是数据集中词 wi 的个数与总词数之比，常数 t 是一个超参数（实验中设为1e-4）
def discard(idx):
    return random.uniform(0, 1) < 1 - math.sqrt(1e-4 / counter[idx_to_token[idx]] * num_tokens)
subsampled_dataset = [[tk for tk in st if not discard(tk)] for st in dataset]
# 二次采样后我们去掉了一半左右的词。
print('# tokens: %d' % sum([len(st) for st in subsampled_dataset])) # # tokens: 375785

# 下面比较一个词在二次采样前后出现在数据集中的次数
def compare_counts(token):
    return print('# %s: before=%d, after=%d' % (token,
                                                sum( [st.count(token_to_idx[token]) for st in dataset]),
                                                sum( [st.count(token_to_idx[token]) for st in subsampled_dataset])
                                                )
                 )
# # the: before=50770, after=2167
compare_counts('the') # 可见高频词“the”的采样率不足1/20

# 低频词“join”则完整地保留了下来
compare_counts('join')  # # join: before=45, after=45

print("----------提取中心词和背景词-------------------")
# 我们将与中心词距离不超过背景窗口大小的词作为它的背景词
"""
下面定义函数提取出所有中心词和它们的背景词。
它每次在整数1和max_window_size（最大背景窗口）之间随机均匀采样一个整数作为背景窗口大小
"""
def get_centers_and_contexts(dataset, max_window_size):
    centers, contexts = [], []
    for st in dataset:
        if len(st) < 2:  # 每个句子至少要有2个词才可能组成一对“中心词-背景词”
            continue
        centers += st
        for center_i in range(len(st)):
            window_size = random.randint(1, max_window_size)
            indices = list(range(max(0, center_i - window_size), min(len(st), center_i + 1 + window_size)))
            indices.remove(center_i)  # 将中心词排除在背景词之外
            contexts.append([st[idx] for idx in indices])
    return centers, contexts

#创建一个数据集：词数分别为7和3的两个句子
tiny_dataset = [list(range(7)), list(range(7, 10))]
print('dataset', tiny_dataset)
# dataset [[0, 1, 2, 3, 4, 5, 6], [7, 8, 9]]
# 设最大背景窗口为2，打印所有中心词和它们的背景词
for center, context in zip(*get_centers_and_contexts(tiny_dataset, 2)):
    print('center', center, 'has contexts', context)

"""
打印中心词下标随机窗口距离的词索引列表
center 0 has contexts [1]
center 1 has contexts [0, 2, 3]
center 2 has contexts [0, 1, 3, 4]
center 3 has contexts [2, 4]
center 4 has contexts [2, 3, 5, 6]
center 5 has contexts [4, 6]
center 6 has contexts [5]
center 7 has contexts [8]
center 8 has contexts [7, 9]
center 9 has contexts [7, 8]
"""
# 我们设最大背景窗口大小为5。下面提取数据集中所有的中心词及其背景词
# all_centers, all_contexts = get_centers_and_contexts(subsampled_dataset, 5)

