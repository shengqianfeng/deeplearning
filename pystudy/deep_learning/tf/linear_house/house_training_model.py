#!/usr/bin/env python
# _*_ coding: utf-8 _*_
# @Time : 2019/12/27 0027 下午 10:17 
# @Author : jeffsmile 
# @File : house_training_model.py
# @desc :房价预测tensorflow的训练模型

import pandas as pd
import numpy as np

def normalize_feature(df):
    """
    数据归一化函数
    :param df:
    :return:
    """
    return df.apply(lambda column: (column - column.mean()) / column.std())

df = normalize_feature(pd.read_csv('data1.csv',
                                   names=['square', 'bedrooms', 'price']))

ones = pd.DataFrame({'ones': np.ones(len(df))})# ones是n行1列的数据框，表示x0恒为1
df = pd.concat([ones, df], axis=1)  # 根据列合并数据
df.head()
# 数据处理：获取 X 和 y
# df.columns[0:3]  ---》Index(['ones', 'square', 'bedrooms'], dtype='object')
X_data = np.array(df[df.columns[0:3]])
y_data = np.array(df[df.columns[-1]]).reshape(len(df), 1)
# (47, 3) <class 'numpy.ndarray'>
print(X_data.shape, type(X_data))
# (47, 1) <class 'numpy.ndarray'>
print(y_data.shape, type(y_data))

print("-------------------------------------------------------------")
# 创建线性回归模型（数据流图）
import tensorflow as tf

alpha = 0.01 # 学习率 alpha
epoch = 500 # 训练全量数据集的轮数

# 输入 X，形状[47, 3]
X = tf.placeholder(tf.float32, X_data.shape)
# 输出 y，形状[47, 1]
y = tf.placeholder(tf.float32, y_data.shape)

# 权重变量 W，形状[3,1]  tf.constant_initializer()默认设置为1
W = tf.get_variable("weights", (X_data.shape[1], 1), initializer=tf.constant_initializer())

# 假设函数 h(x) = w0*x0+w1*x1+w2*x2, 其中x0恒为1
# 推理值 y_pred  形状[47,1]
y_pred = tf.matmul(X, W)

# 损失函数采用最小二乘法，y_pred - y 是形如[47, 1]的向量。
# tf.matmul(a,b,transpose_a=True) 表示：矩阵a的转置乘矩阵b，即 [1,47] X [47,1]
# 损失函数操作 loss
loss_op = 1 / (2 * len(X_data)) * tf.matmul((y_pred - y), (y_pred - y), transpose_a=True)
# 随机梯度下降优化器 opt
opt = tf.train.GradientDescentOptimizer(learning_rate=alpha)
# 单轮训练操作 train_op  最小化损失函数
train_op = opt.minimize(loss_op)

# 创建会话（运行环境）
with tf.Session() as sess:
    # 初始化全局变量
    sess.run(tf.global_variables_initializer())
    # 开始训练模型，
    # 因为训练集较小，所以每轮都使用全量数据训练
    for e in range(1, epoch + 1):
        # 执行train_op背后的过程是数据流图会把当前的训练数据训练一遍得出w并求出梯度，然后更新w参数。
        sess.run(train_op, feed_dict={X: X_data, y: y_data})
        # 每10个epoch输出当前损失值和权重
        if e % 10 == 0:
            loss, w = sess.run([loss_op, W], feed_dict={X: X_data, y: y_data})
            log_str = "Epoch %d \t Loss=%.4g \t Model: y = %.4gx1 + %.4gx2 + %.4g"
            print(log_str % (e, loss, w[1], w[2], w[0]))

            # Epoch 500 	 Loss=0.132 	 Model: y = 0.8304x1 + 0.0008239x2 + 4.138e-09



