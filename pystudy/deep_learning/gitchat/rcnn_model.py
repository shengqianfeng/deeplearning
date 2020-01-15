#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
@File : rcnn_model.py
@Author : jeffsheng
@Date : 2020/1/7 0007
@Desc : rcnn模型
"""
import io
import os
from zipfile import ZipFile

import numpy as np
import requests
import tensorflow as tf

from pystudy.config.log_config import log_base_config

logger = log_base_config.get_log("intent_train")



class rcnn(object):

    def __init__(self,model_output_path):
        # 设定 RNN 模型的参数
        self.epochs = 60
        self.batch_size = tf.placeholder(dtype=tf.int32)
        # 每个文本的最大长度为 25 个单词，这样会将较长的文本剪切为 25 个，不足的用零填充
        self.max_sequence_length = 25
        # 表示rnn隐藏层神经元的个数，注意并不是rnn网络展开的时间步数，是rnn每个cell单元的神经元个数，rnn的cell单元可以当成全连接层来理解
        self.rnn_size = 10
        # 每个单词都将被嵌入到一个长度为 50 的可训练向量中
        self.embedding_size = 50
        # 单词被收录到此表中的最小词频
        self.min_word_frequency = 10
        self.learning_rate = 0.001
        self.dropout_keep_prob = tf.placeholder(tf.float32)
        # 模型保存路径
        self.model_output_path = model_output_path
        #  inp大小为 [None, max_sequence_length]，max_sequence_length就是每个文本句子的组成单词数量
        self.inp = tf.placeholder(tf.int32, [None, self.max_sequence_length], name='inp')
        # 是一个整数，值为 0 或 1, 分别表示 ham 和 spam
        self.labels = tf.placeholder(tf.int32, [None], name='labels')
        # 词向量的大小
        self.n_words = None

        self.train_step = None
        self.optimizer = None
        self.accuracy = None
        self.loss = None
        self.sess = None
        self.soft_max_y=None
        self.predict_result=None


    def build_model(self):
        with tf.name_scope('simple-rcnn'):
            if tf.__version__[0] >= '1':
                cell = tf.contrib.rnn.BasicRNNCell(num_units=self.rnn_size)
            else:
                cell = tf.nn.rnn_cell.BasicRNNCell(num_units=self.rnn_size)

            # 初始化词嵌入矩阵
            # W可以理解为：每个词向量中的词都有对应大小为embedding_size=50的向量，这里值为随机值区间(-1,1)
            self.W  = tf.Variable(tf.random_uniform([self.n_words, self.embedding_size], -1.0, 1.0), name='W')
            # embedding_output理解为： 将inp中每个单词的整数索引，映射到这个可训练的嵌入矩阵embedding_mat的某一行
            """
            比如：句子：i love python 
                  索引向量（inp）：[100,  555  ,987]
                  embedding_output结果为：[[.......],[.......],[..........]],其中的每个[.......]就是句子单词对应向量索引查找到的词嵌入矩阵行，因为词嵌入矩阵是每个单词的embedding_size扩展表示
            再比如：
            w = tf.constant([[1,2],[3,4],[5,6]])
            inp=tf.constant([[1,1],[0,1]])
            b = tf.nn.embedding_lookup(w, inp)
            sess.run(b)的结果是：[[[3 4]
                                  [3 4]]
                                
                                 [[1 2]
                                  [3 4]]]
            """
            # 输出格式： (？，max_sequence_length，embedding_size)
            self.embedding_output = tf.nn.embedding_lookup(self.W, self.inp)

            # 用 tf.nn.dynamic_rnn 建立 RNN 序列
            """
            tf.nn.dynamic_rnn:
            1 单个的 RNNCell，调用一次就只是在序列时间上前进了一步。 
              所以需要使用 tf.nn.dynamic_rnn 函数，它相当于调用了n次 RNNCell,即通过 {h0,x1, x2, …., xn} 直接得到 {h1,h2…,hn},{output1,output2…,outputn}
            2 tf.nn.dynamic_rnn 以前面定义的 cell 为基本单元建立一个展开的RNN序列网络,将词嵌入和初始状态输入其中,返回了 output，还有最后的 state。
            3 output是一个三维的tensor，是time_steps步的所有的输出，形状为 (batch_size, time_steps, cell.output_size)
               state是最后一步的隐状态，形状为 (batch_size, cell.state_size)
            """
            output, state = tf.nn.dynamic_rnn(cell, inputs=self.embedding_output, dtype=tf.float32)

        # 再为 RNN 添加 dropout
        '''
        tf.nn.dropout 用来减轻过拟合
        dropout_keep_prob 是保留比例，是神经元被选中的概率，和输入一样，也是一个占位符，取值为 (0,1] 之间
        '''
        output = tf.nn.dropout(output, self.dropout_keep_prob)
        '''
        tf.transpose 用于将张量进行转置，张量的原有维度顺序是 [0, 1, 2], 则 [1, 0, 2] 是告诉 tf 要将 0 和 1 维转置
         0 代表三维数组的高，1 代表二维数组的行，2 代表二维数组的列。 
         即将输出 output 的维度 [batch_size, time_steps, cell.output_size] 
         变成 [time_steps, batch_size, cell.output_size]，表示每个时间步的batch_size份的cell单元的输出：这里为(25，？，10)，这里的10就是rcnn_size
        '''
        output = tf.transpose(output, [1, 0, 2])
        # 切掉最后一个时间步的输出作为预测值 tf.gather 用于将向量中某些索引值提取出来，得到新的向量。
        last = tf.gather(output, int(output.get_shape()[0]) - 1)

        # 将 output 传递给一个全连接层，来得到 logits_out
        '''
        为了完成 RNN 的分类预测，通过一个全连接层 将 rnn_size 长度的输出变为二分类输出
        在 RNN 中，全连接层可以将 embedding 空间拉到隐层空间，将隐层空间转回 label 空间
        
        解读tf.truncated_normal函数：
            用来从截断的正态分布中随机抽取值，即生成的值服从指定平均值和标准偏差的正态分布， 
        但是如果生成的值与均值的差值大于两倍的标准差，即在区间（μ-2σ，μ+2σ）之外，
        则丢弃并重新进行选择，这样可以保证生成的值都在均值附近
        其中参数 shape 表示生成的张量的维度是 [rnn_size, 2]， mean 均值默认为 0， stddev 标准差设置为 0.1。
        '''
        with tf.name_scope('weights'):
            weight = tf.Variable(tf.truncated_normal([self.rnn_size, 2], stddev=0.1))
        with tf.name_scope('biases'):
            bias = tf.Variable(tf.constant(0.1, shape=[2]))
        # logits 是这个全连接层的输出，作为 softmax 的输入，在接下来定义损失函数时用到，logits_out形状（？，2）
            self.logits_out = tf.matmul(last, weight) + bias
            self.soft_max_y = tf.nn.softmax(self.logits_out)
            # 返回每行最大值的下标
            self.predict_result = tf.argmax(self.soft_max_y, 1, name='predictions')
        # 定义损失函数和准确率函数
        '''
        tf.nn.sparse_softmax_cross_entropy_with_logits:
        sparse_softmax_cross_entropy_with_logits(
            _sentinel=None,
            labels=None,
            logits=None,
            name=None
        )
        logits：是神经网络最后一层的输出，如果有 batch 的话，形状是 [batch_size, num_classes]
        labels：是实际的标签，形状是 [batch_size, 1],每个 label 的取值是 [0, num_classes) 的离散值，是哪一类就标记哪个类对应的 label

            这个函数将 softmax 和 cross_entropy 放在一起计算， 先对网络最后一层的输出做一个 softmax 求取输出属于某一类的概率， 
        然后 softmax 的输出向量再和样本的实际标签做一个交叉熵 cross_entropy，来计算的两个概率分布之间的距离，这是分类问题中使用比较广泛的损失函数之一
        '''
        with tf.name_scope('loss'):
            losses = tf.nn.sparse_softmax_cross_entropy_with_logits(logits=self.logits_out, labels=self.labels)

        # 上面损失函数的返回值是一个向量，是一个 batch 中每个样本的 loss，不是一个数， 需要通过 tf.reduce_mean 对向量求均值，计算 batch 内的平均 loss。
            self.loss = tf.reduce_mean(losses)

        # 定义准确率函数
        '''
        tf.argmax 用来返回最大值 1 所在的索引位置，因为标签向量是由 0,1 组成，因此返回了预测类别标签， 
        再用 tf.equal 来检测预测与真实标签是否匹配，返回一个布尔数组,
        用 tf.cast 将布尔值转换为浮点数 [1,0,1,1...] 最后用 tf.reduce_mean 计算出平均值即为准确率
        '''
        with tf.name_scope('accuracy'):
            self.accuracy = tf.reduce_mean(tf.cast(tf.equal(tf.argmax(self.logits_out, 1), tf.cast(self.labels, tf.int64)), tf.float32))

        # 选择优化算法
        '''
        RMSProp 是 Geoff Hinton 提出的一种自适应学习率方法，为了解决 Adagrad 学习率急剧下降问题的,
        这种方法很好的解决了深度学习中过早结束的问题，适合处理非平稳目标，对于RNN效果很好
        '''
        self.optimizer = tf.train.RMSPropOptimizer(self.learning_rate)
        self.train_step = self.optimizer.minimize(self.loss)

    def train_model(self,x_train,y_train,x_test,y_test):
        with tf.Session() as self.sess:
            self.sess.run(tf.initialize_all_variables())
            saver = tf.train.Saver()
            train_loss = []
            test_loss = []
            train_accuracy = []
            test_accuracy = []
            '''
            训练的流程为：
            1 Shuffle 训练数据：在每一代 epoch 遍历时都 shuffle 一下数据可以避免过度训练
            2 选择训练集, 训练每个 batch
            3 计算训练集和测试集的损失 loss 和准确率 accuracy
            '''
            user_batch_size =150
            # 开始训练
            for epoch in range(self.epochs):

                # Shuffle 训练集
                shuffled_ix = np.random.permutation(np.arange(len(x_train)))
                x_train = x_train[shuffled_ix]
                y_train = y_train[shuffled_ix]
                # 用 Mini batch 梯度下降法
                num_batches = int(len(x_train) / user_batch_size) + 1

                for i in range(num_batches):
                    # 选择每个 batch 的训练数据
                    min_ix = i * user_batch_size
                    max_ix = np.min([len(x_train), ((i + 1) * user_batch_size)])
                    x_train_batch = x_train[min_ix:max_ix]
                    y_train_batch = y_train[min_ix:max_ix]

                    # 进行训练： 用 Session 来 run 每个 batch 的训练数据，逐步提升网络的预测准确性
                    train_dict = {self.inp: x_train_batch, self.labels: y_train_batch, self.dropout_keep_prob: 0.5}
                    self.sess.run(self.train_step, feed_dict=train_dict)

                # 将训练集每一代的 loss 和 accuracy 加到整体的损失和准确率中去
                temp_train_loss, temp_train_acc = self.sess.run([self.loss, self.accuracy], feed_dict=train_dict)
                train_loss.append(temp_train_loss)
                train_accuracy.append(temp_train_acc)
                # 同时计算并记录测试集每一代的损失和准确率
                test_dict = {self.inp: x_test, self.labels: y_test, self.dropout_keep_prob: 1.0}
                temp_test_loss, temp_test_acc = self.sess.run([self.loss, self.accuracy], feed_dict=test_dict)
                test_loss.append(temp_test_loss)
                test_accuracy.append(temp_test_acc)
                print('Epoch: {}, Test Loss: {:.2}, Test Acc: {:.2}'.format(epoch + 1, temp_test_loss, temp_test_acc))

            # 将模型保存到save/model.ckpt文件
            saver_path = saver.save(self.sess, self.model_output_path)
            logger.info("Final Model saved in file: %s" % saver_path)
        return train_loss,test_loss,train_accuracy,test_accuracy


    def load_data(self):
        data_dir = 'temp'
        data_file = 'text_data.txt'
        if not os.path.exists(data_dir):
            os.makedirs(data_dir)

        # 直接下载zip格式的数据集
        if not os.path.isfile(os.path.join(data_dir, data_file)):
            zip_url = 'http://archive.ics.uci.edu/ml/machine-learning-databases/00228/smsspamcollection.zip'
            r = requests.get(zip_url)
            z = ZipFile(io.BytesIO(r.content))
            file = z.read('SMSSpamCollection')

            # 格式化数据
            text_data = file.decode()
            text_data = text_data.encode('ascii', errors='ignore')
            text_data = text_data.decode().split('\n')

            # 将数据存储到 text 文件
            with open(os.path.join(data_dir, data_file), 'w') as file_conn:
                for text in text_data:
                    file_conn.write("{}\n".format(text))
        else:
            # 从 text 文件打开数据
            text_data = []
            with open(os.path.join(data_dir, data_file), 'r') as file_conn:
                for row in file_conn:
                    text_data.append(row)
            text_data = text_data[:-1]

        # 数据预处理
        # 首先，将数据中的标签和邮件文本分开，得到 text_data_target 和 text_data_train
        text_data = [x.split('\t') for x in text_data if len(x) >= 1]
        [text_data_target, text_data_train] = [list(x) for x in zip(*text_data)]
        return text_data_target,text_data_train