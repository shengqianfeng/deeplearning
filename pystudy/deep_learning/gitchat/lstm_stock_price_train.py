#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
@File : lstm_stock_price_train.py
@Author : jeffsheng
@Date : 2020/1/17
@Desc : tensorflow1.x版本的股票价格预测训练模型
"""

import numpy as np
import pandas as pd
import math
import sklearn
import sklearn.preprocessing
import datetime
import os
import matplotlib.pyplot as plt
import tensorflow as tf


# 将 default graph 重新初始化，保证内存中没有其他的 Graph，相当于清空所有的张量
tf.reset_default_graph()

# 为 graph 建立会话 session
sess = tf.Session()
print("-----------导入所有股票价格数据，进行可视化----------")
"""
每一条数据是某一天某个公司的 open、close、low、high 价格和当天交易量 volume
symbol 是公司的代号
volume 是成交量
"""
# 这个数据集有 501 家公司从 2016-01-05 到 2016-12-30 的股票价格
df = pd.read_csv("./temp/prices-split-adjusted.csv", index_col = 0)
df.info()
df.tail()
df.describe()
print('\nnumber of different stocks: ', len(list(set(df.symbol))))        # 不同股票的个数
print(list(set(df.symbol))[:10])

# 将'EQIX' 的五个数据进行可视化
plt.figure(figsize=(15, 5));
plt.subplot(1,2,1);
plt.plot(df[df.symbol == 'EQIX'].open.values, color='red', label='open')
plt.plot(df[df.symbol == 'EQIX'].close.values, color='green', label='close')
plt.plot(df[df.symbol == 'EQIX'].low.values, color='blue', label='low')
plt.plot(df[df.symbol == 'EQIX'].high.values, color='yellow', label='high')
plt.title('stock price')
plt.xlabel('time [days]')
plt.ylabel('price')
plt.legend(loc='best')

plt.subplot(1,2,2);
plt.plot(df[df.symbol == 'EQIX'].volume.values, color='black', label='volume')
plt.title('stock volume')
plt.xlabel('time [days]')
plt.ylabel('volume')
plt.legend(loc='best')
plt.show()

print("------------数据预处理-------------------")
def normalize_data(df):
    """
    1 将数据集的 open high low close 四列数据进行 MinMaxScaler 标准化
    2 MinMaxScaler 是一种标准化方法，用来将特征的数值缩放到指定的最小值到最大值之间，fit_transform 将填充数据并对数据进行此转换
    3 常用的是将数据缩放到 0~1 之间。通过标准化处理，可以使各个特征具有相同的尺度，在训练神经网络时能够加速权重参数的收敛
    :param df:
    :return:
    """
    min_max_scaler = sklearn.preprocessing.MinMaxScaler()
    df['open'] = min_max_scaler.fit_transform(df.open.values.reshape(-1,1))
    df['high'] = min_max_scaler.fit_transform(df.high.values.reshape(-1,1))
    df['low'] = min_max_scaler.fit_transform(df.low.values.reshape(-1,1))
    df['close'] = min_max_scaler.fit_transform(df['close'].values.reshape(-1,1))
    return df

print("----------将数据集分为训练集测试集----------------")
# 设定数据集划分的比例为80% 的训练集，20% 的测试集
train_set_size_percentage = 80
def load_data(stock, seq_len):
    # 将数据转化为 numpy array
    data_raw = stock.as_matrix()
    data = []

    # 构建数据 data: 用长为 seq_len 的窗口，从头截取 data_raw 中的数据，每一段为 data 中的一条样本
    for index in range( len(data_raw) - seq_len ):
        data.append( data_raw[index: index + seq_len] )

    data = np.array(data);

    # 根据设置的比例计算 训练集，测试集 的大小
    train_set_size = int( np.round( train_set_size_percentage / 100 * data.shape[0] ) );

    # data 中从开始到 train_set_size 这些是 训练集，剩下那些是测试集
    x_train = data[:train_set_size, :-1, :]
    y_train = data[:train_set_size, -1, :]

    x_test = data[train_set_size:, :-1, :]
    y_test = data[train_set_size:, -1, :]

    return [x_train, y_train, x_test, y_test]


# 删除公司和成交量两列，只留下 ['open', 'close', 'low', 'high'] 四列
df_stock = df[df.symbol == 'EQIX'].copy()
df_stock.drop(['symbol'], 1, inplace = True)
df_stock.drop(['volume'], 1, inplace = True)

cols = list(df_stock.columns.values)
print('df_stock.columns.values = ', cols)
# 对选定数据的四列进行min-max normalization
df_stock_norm = df_stock.copy()
df_stock_norm = normalize_data(df_stock_norm)
# print(df_stock_norm)

# 接着设定窗口序列长度为 20，生成 训练集，测试集的 x 和 y
seq_len = 20
x_train, y_train, x_test, y_test = load_data(df_stock_norm, seq_len)

print('x_train.shape = ',x_train.shape)
print('y_train.shape = ', y_train.shape)
print('x_test.shape = ', x_test.shape)
print('y_test.shape = ',y_test.shape)

# 可视化处理后的数据
plt.figure(figsize=(15, 5));
plt.plot(df_stock_norm.open.values, color='red', label='open')
plt.plot(df_stock_norm.close.values, color='green', label='low')
plt.plot(df_stock_norm.low.values, color='blue', label='low')
plt.plot(df_stock_norm.high.values, color='yellow', label='high')
plt.title('stock')
plt.xlabel('time [days]')
plt.ylabel('normalized price/volume')
plt.legend(loc='best')
plt.show()

print("---------对训练集数据做 shuffle----------")
perm_array  = np.arange(x_train.shape[0])
np.random.shuffle(perm_array)

print("--------建立 LSTM 模型---------------")
index_in_epoch = 0;


# 获取每一批的训练数据
def get_next_batch(batch_size):
    global index_in_epoch, x_train, perm_array
    # 由 start 和 end 截取 shuffle 后的数据 perm_array， 来得到每一批数据
    start = index_in_epoch
    # 每次获取完一批后，index 向后更新一次
    index_in_epoch += batch_size
    # 如果 end 超过了训练集数量，就重新 shuffle 数据，从头开始取 batch
    if index_in_epoch > x_train.shape[0]:
        np.random.shuffle(perm_array)
        start = 0
        index_in_epoch = batch_size
    end = index_in_epoch
    return x_train[perm_array[start:end]], y_train[perm_array[start:end]]


print("----------定义模型参数---------------------")
"""
n_steps :  根据窗口的设定，每个训练数据是长度等于 seq_len - 1 的序列，所以展开 LSTM n_steps ＝ seq_len - 1 个时间步
n_inputs : 每个 input 有 4 个特征，即在每个时刻有 n_inputs＝ 4 个值，open high low close
num_units: 为 LSTM cell 中 h 和 c 的维度
n_outputs : 每个预测结果也是包含 n_outputs ＝ 4 个值
"""
n_steps = seq_len - 1
n_inputs = 4
num_units = 200
n_outputs = 4

learning_rate = 0.001
batch_size = 50
n_epochs = 100
train_set_size = x_train.shape[0]
test_set_size = x_test.shape[0]
# 定义 x y 的 placeholder
X = tf.placeholder(tf.float32, [None, n_steps, n_inputs])
y = tf.placeholder(tf.float32, [None, n_outputs])
# 建立 LSTM
cell = tf.contrib.rnn.BasicLSTMCell(num_units = num_units, activation = tf.nn.elu)
rnn_outputs, states = tf.nn.dynamic_rnn(cell, X, dtype = tf.float32)
# 输出outputs
# 将上一步 LSTM 的输出 rnn_outputs 的维度从 [batch_size, n_steps, num_units] 变换到 [batch_size * n_steps, num_units]
# 然后用一个全连接层，它的输出大小和预测输出值维度相等 ＝ n_outputs，这样输出的维度为 [batch_size * n_steps, n_outputs]
# 接着再把这个 tensor 变为 [batch_size, n_steps, n_outputs]
# 最后只保留 outputs 的最后一个时间步 n_steps - 1 时的输出
stacked_rnn_outputs = tf.reshape(rnn_outputs, [-1, num_units])
stacked_outputs = tf.layers.dense(stacked_rnn_outputs, n_outputs)
outputs = tf.reshape(stacked_outputs, [-1, n_steps, n_outputs])
outputs = outputs[:,n_steps-1,:]

print("---------- 定义 Loss 和 optimizer ------------")
# 损失函数为 MSE，优化算法为 AdamOptimizer
loss = tf.reduce_mean(tf.square(outputs - y))
optimizer = tf.train.AdamOptimizer(learning_rate=learning_rate)
training_op = optimizer.minimize(loss)

train_loss = []
test_loss = []

print("----------------运行计算图--------------------")
with tf.Session() as sess:
    # 初始化变量
    sess.run(tf.global_variables_initializer())

    for epoch in range(n_epochs):
        num_batches = int(train_set_size / batch_size) + 1
        for i in range(num_batches):
            # 获得每一批的 x 和 y
            x_batch, y_batch = get_next_batch(batch_size)
            # 训练模型时，将 训练集 x 和 y 喂给网络，然后用优化算法训练
            sess.run(training_op, feed_dict={X: x_batch, y: y_batch})

        # 计算 训练集 和 测试集的 MSE
        mse_train = loss.eval(feed_dict={X: x_train, y: y_train})
        mse_test = loss.eval(feed_dict={X: x_test, y: y_test})

        train_loss.append(mse_train)
        test_loss.append(mse_test)

        if epoch % 10 == 0:
            print('Epoch: {}, MSE train: {:.6}, MSE test: {:.6}'.format(epoch+1, mse_train, mse_test))

    # 当输入数据集分别为 训练集，测试集 时，得到相应的预测值
    y_train_pred = sess.run(outputs, feed_dict={X: x_train})
    y_test_pred = sess.run(outputs, feed_dict={X: x_test})


print("--------------------------------可视化结果-------------------------------------")
# ft：代表预测值y的第几列: 0 = open, 1 = close, 2 = highest, 3 = lowest
ft = 0
plt.figure(figsize=(15, 5));
plt.subplot(1,2,1);
# 画出训练集的实际值和预测值
plt.plot(np.arange(y_train.shape[0]), y_train[ : , ft], color='blue', label='train target')
plt.plot(np.arange(y_train_pred.shape[0]), y_train_pred[ : , ft], color='red', label='train prediction')

# 画出测试集的实际值和预测值
plt.plot(np.arange(y_train.shape[0], y_train.shape[0] + y_test.shape[0]), y_test[ : , ft], color='gray', label='test target')
plt.plot(np.arange(y_train_pred.shape[0], y_train_pred.shape[0]+y_test_pred.shape[0]), y_test_pred[:,ft], color='green', label='test prediction')

plt.title('past and future stock prices')
plt.xlabel('time [days]')
plt.ylabel('normalized price')
plt.legend(loc='best');

plt.subplot(1,2,2);

# 只看测试集的实际值和预测值
plt.plot(np.arange(y_train.shape[0],y_train.shape[0] + y_test.shape[0]),
                y_test[:,ft], color='grey', label='test target')
plt.plot(np.arange(y_train_pred.shape[0], y_train_pred.shape[0] + y_test_pred.shape[0]),
         y_test_pred[:,ft], color='green', label='test prediction')

plt.title('future stock prices')
plt.xlabel('time [days]')
plt.ylabel('normalized price')
plt.legend(loc='best');
plt.show()
