#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
@File : sequential.py
@Author : jeffsheng
@Date : 2020/1/3
@Desc : 用Sequential类来实现前面structural_model.py描述的MLP类，并使用随机初始化的模型做一次前向计算

"""
import tensorflow as tf
print(tf.__version__)


"""
Sequential来简化模型定义：
优点：
1 Sequential类继承自tf.keras.Model类
2 当模型的前向计算为简单串联各个层的计算时，可以通过更加简单的方式定义模型。这正是Sequential类的目的.
3 Sequential提供add函数来逐一添加串联的Block子类实例，而模型的前向计算就是将这些实例按添加的顺序逐一计算
缺点：
4 虽然Sequential类可以使模型构造更加简单，且不需要定义call函数，但直接继承tf.keras.Model类可以极大地拓展模型构造的灵活性
"""
model = tf.keras.models.Sequential([
    tf.keras.layers.Flatten(),
    tf.keras.layers.Dense(256, activation=tf.nn.relu),
    tf.keras.layers.Dense(10),
])
X = tf.random.uniform((2,20))
"""
tf.Tensor(
[[ 0.19341628  0.25244334 -0.10456292  0.3209575   0.3480761  -0.2373815
  -0.0703845  -0.16567561 -0.02303264 -0.1222413 ]
 [ 0.250063   -0.00751293 -0.24290471  0.29253864  0.4299389  -0.06697307
   0.10469621 -0.16417655 -0.3520375  -0.00803327]], shape=(2, 10), dtype=float32)
"""
print(model(X))


