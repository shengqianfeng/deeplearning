#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
@File : predict_convnet.py
@Author : jeffsheng
@Date : 2019/12/10
@Desc : 对训练好的CNN进行结果预测
"""
from PIL import Image
from pystudy.nn_study.cnn.simple_convNet import SimpleConvNet
import numpy as np

network = SimpleConvNet()
network.load_params("params.pkl")
# 打开一个jpg图像文件，注意是当前路径:
img = Image.open('img/5.jpg')
im2 = np.array(img)
im = im2.reshape(1,1,28,28)
res = network.predict(im)
print(res)
print("图像结果为：", np.argmax(res))



