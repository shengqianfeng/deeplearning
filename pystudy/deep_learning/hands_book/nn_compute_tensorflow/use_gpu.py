#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
@File : use_gpu.py
@Author : jeffsheng
@Date : 2020/1/3
@Desc : 介绍如何使用单块NVIDIA GPU来计算
"""

import tensorflow as tf
import numpy as np
print(tf.__version__)

gpus = tf.config.experimental.list_physical_devices(device_type='GPU')
cpus = tf.config.experimental.list_physical_devices(device_type='CPU')
print("可用的GPU：",gpus,"\n可用的CPU：", cpus)


print("--------check available device---------")
from tensorflow.python.client import device_lib
print(device_lib.list_local_devices())


print("----------使用tf.device()来指定特定设备(GPU/CPU)----")
with tf.device('GPU:0'):
    a = tf.constant([1,2,3],dtype=tf.float32)
    b = tf.random.uniform((3,))
    print(tf.exp(a + b) * 2)





