# coding: utf-8
import sys, os
# sys.path.append(os.pardir)  # 为了导入父目录的文件而进行的设定
import numpy as np
from pystudy.dataset.mnist import load_mnist
from PIL import Image
# 展示minist训练集的第一张数字图像

def img_show(img):
    #把保存为NumPy数组的图像数据转换为PIL用的数据对象
    pil_img = Image.fromarray(np.uint8(img))
    pil_img.show()
# x_train：60000张图 t_train：6000个结果一一对应
(x_train, t_train), (x_test, t_test) = load_mnist(flatten=True, normalize=False)

img = x_train[0]
label = t_train[0]
print(label)  # 5

print(img.shape)  # (784,)
# reshape()方法的参数指定期望的形状
img = img.reshape(28, 28)  # 把图像的形状变为原来的尺寸
print(img.shape)  # (28, 28)
# 将数据集的第一张图片从数字还原为图像
img_show(img)
