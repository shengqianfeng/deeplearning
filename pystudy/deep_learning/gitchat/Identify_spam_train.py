#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
@File : Identify_spam_train.py
@Author : jeffsheng
@Date : 2020/1/7
@Desc : rnn识别垃圾邮件应用
"""
# 加载需要的包
import os
import re

import matplotlib.pyplot as plt
import numpy as np
import tensorflow as tf
from tensorflow.python.framework import ops
from  pystudy.sysutils.data_utils import clean_text
from pystudy.config.log_config import log_base_config
from pystudy.deep_learning.gitchat.rcnn_model import rcnn

logger = log_base_config.get_log("intent_train")
# 将 default graph 重新初始化，保证内存中没有其他的 Graph，相当于清空所有的张量
ops.reset_default_graph()

# 为 graph 建立会话 session
sess = tf.Session()

output_path = './model/classifier_save/normal/'
if not os.path.exists(output_path):
    os.makedirs(output_path)

model_output_path = os.path.join(output_path, "model.ckpt")


net = rcnn(model_output_path)
# 导入数据
text_data_target,text_data_train = net.load_data()



# 调用 clean_text 清理文本
text_data_train = [clean_text(x) for x in text_data_train]


# 接下来将文本转为词的 ID 序列
# 用 TensorFlow 中一个内置的 VocabularyProcessor 函数来处理文本
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
vocab_processor = tf.contrib.learn.preprocessing.VocabularyProcessor(net.max_sequence_length, min_frequency=net.min_word_frequency)
text_processed = np.array(list(vocab_processor.fit_transform(text_data_train)))
# 词向量的大小
vocab_size = len(vocab_processor.vocabulary_)

import pickle
with open('./vocab/word2id.pkl', 'wb') as f:
    pickle.dump(vocab_processor.vocabulary_, f)

print("Vocabulary Size: {:d}".format(vocab_size))

net.n_words = vocab_size

# shuffle，可以打乱数据行序，使数据随机化
text_processed = np.array(text_processed)
# 将text_data_target转为0和1的结果标签
text_data_target = np.array([1 if x == 'ham' else 0 for x in text_data_target])
'''
shuffle与permutation的区别
    1函数shuffle与permutation都是对原来的数组进行重新洗牌（即随机打乱原来的元素顺序）；
    2区别在于shuffle直接在原来的数组上进行操作，改变原来数组的顺序，无返回值。
    3而permutation不直接在原来的数组上进行操作，而是返回一个新的打乱顺序的数组，并不改变原来的数组
'''
# shuffled_ix = np.random.permutation(np.arange(len(text_data_target)))
# 按照随机打乱词向量及标签的顺序重新返回数据集和训练集
# x_shuffled = text_processed[shuffled_ix]
# y_shuffled = text_data_target[shuffled_ix]
x_shuffled = text_processed
y_shuffled = text_data_target
'''
shuffle 数据后，将数据集分为 80% 训练集和 20% 测试集 
如果想做交叉验证 cross-validation ，可以将 测试集 进一步分为测试集和验证集来调参
'''
ix_cutoff = int(len(y_shuffled)*0.80)
# 分割数据：训练集和测试集
x_train, x_test = x_shuffled[:], x_shuffled[:]
# 分割标签：训练集标签和测试集标签
y_train, y_test = y_shuffled[:], y_shuffled[:]
print("80-20 Train Test split: {:d} -- {:d}".format(len(y_train), len(y_test)))

# 将输入数据做词嵌入 Embedding


# 接下来，建立 embedding
# 用 embedding 将索引转化为特征，这是一种特征学习方法，可以用来自动学习数据集中各个单词的显著特征,
# 它可以将单词映射到固定长度为 embedding_size 大小的向量
"""
建立嵌入层的步骤：
1 首先建立一个嵌入矩阵 embedding_mat，它是一个 tensor 变量，大小为 [vocab_size × embedding_size]，
将它的元素随机初始化为 [-1, 1] 之间
    解释：vocab_size表示文本中所有不重复单词总数，embedding_size表示每个单词的向量长度
2 然后用 tf.nn.embedding_lookup 函数来找嵌入矩阵 embedding_mat 中与 x_data 的每个元素所对应的行，
也就是将每个单词的整数索引，映射到这个可训练的嵌入矩阵 embedding_mat 的某一行
"""


# 建立RNN模型
# 定义 RNN cell
"""
BasicRNNCell 是 RNN 的基础类， 激活函数默认是 tanh， 每调用一次，就相当于在时间上“推进了一步”， 
它的参数 num_units 就是隐藏层神经元的个数

RNNCell解释：
    BasicRNNCell 和 BasicLSTMCell 都是抽象类 RNNCell 的两个子类， 每个 RNNCell 都有一个 call 方法，
使用为 :   (output, next_state) = call(input, state)
其中输入数据input的形状为 (batch_size, input_size)， 得到的隐层状态形状 (batch_size, state_size)，输出形状为 (batch_size, output_size)
比如：输入一个初始状态 h0 和输入 x1，调用 call(x1, h0) 后就可以得到 (output1, h1)， 再调用一次 call(x2, h1) 就可以得到 (output2, h2)
"""
print("-------------------建立RNN模型-----------------------------")
net.build_model()

print("--------------训练模型-------------")
train_loss,test_loss,train_accuracy,test_accuracy = net.train_model(x_train,y_train,x_test,y_test)

print("------------可视化损失函数 loss 和准确率 accuracy-----------")
# 损失随着时间的变化
epoch_seq = np.arange(1, net.epochs+1)
plt.plot(epoch_seq, train_loss, 'k--', label='Train Set')
plt.plot(epoch_seq, test_loss, 'r-', label='Test Set')
plt.title('Softmax Loss')
plt.xlabel('Epochs')
plt.ylabel('Softmax Loss')
plt.legend(loc='upper left')
plt.show()

# 准确率随着时间的变化
plt.plot(epoch_seq, train_accuracy, 'k--', label='Train Set')
plt.plot(epoch_seq, test_accuracy, 'r-', label='Test Set')
plt.title('Test Accuracy')
plt.xlabel('Epochs')
plt.ylabel('Accuracy')
plt.legend(loc='lower right')
plt.show()

