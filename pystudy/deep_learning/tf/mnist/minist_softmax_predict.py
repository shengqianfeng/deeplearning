#!/usr/bin/env python
# _*_ coding: utf-8 _*_
# @Time : 2019/12/28 0028 上午 11:59 
# @Author : jeffsmile 
# @File : minist_softmax_predict.py
# @desc : tensorflow预测softmax训练模型

import os
from keras.models import load_model
from PIL import Image
import numpy as np


save_dir = "./model/"
model_name = 'keras_mnist.h5'
model_path = os.path.join(save_dir, model_name)
mnist_model = load_model(model_path)

img = Image.open('img/1.jpg')
im2 = np.array(img)
im = im2.reshape(1,784)
predicted_classes = mnist_model.predict_classes(im)
print(predicted_classes)