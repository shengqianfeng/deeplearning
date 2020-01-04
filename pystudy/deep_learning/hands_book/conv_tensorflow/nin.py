#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
@File : nin.py
@Author : jeffsheng
@Date : 2020/1/3 0003
@Desc : NIN
思路：
    1 串联多个由[卷积层+“全连接”层]构成的小网络来构建一个深层网络
    2 如果想在全连接层后再接上卷积层，则需要将全连接层的输出变换为四维
        1×1卷积层可以看成全连接层，其中空间维度（高和宽）上的每个元素相当于样本，通道相当于特征
    3 NiN使用1×1卷积层来替代全连接层，从而使空间信息能够自然传递到后面的层中去

小结：
1 NiN重复使用由卷积层和代替全连接层的1×1卷积层构成的NiN块来构建深层网络
2 NiN去除了容易造成过拟合的全连接输出层，而是将其替换成输出通道数等于标签类别数的NiN块和全局平均池化层。
3 NiN的以上设计思想影响了后面一系列卷积神经网络的设计
"""
import tensorflow as tf
print(tf.__version__)

for gpu in tf.config.experimental.list_physical_devices('GPU'):
    tf.config.experimental.set_memory_growth(gpu, True)


# nin块的定义
"""
NiN块是NiN中的基础块。
它由1个卷积层加2个充当全连接层的1×1卷积层串联而成.
其中第一个卷积层的超参数可以自行设置，而第二和第三个卷积层的超参数一般是固定的
"""
def nin_block(num_channels, kernel_size, strides, padding):
    blk = tf.keras.models.Sequential()
    blk.add(tf.keras.layers.Conv2D(num_channels, kernel_size,
                                   strides=strides, padding=padding, activation='relu'))
    blk.add(tf.keras.layers.Conv2D(num_channels, kernel_size=1,activation='relu'))
    blk.add(tf.keras.layers.Conv2D(num_channels, kernel_size=1,activation='relu'))
    return blk
"""
NIN网络结构设计：
   1 NiN是在AlexNet问世不久后提出的。它们的卷积层设定有类似之处
   2 NiN使用卷积窗口形状分别为11×11、5×5和3×3的卷积层，相应的输出通道数也与AlexNet中的一致。
   3 每个NiN块后接一个步幅为2、窗口形状为3×3的最大池化层
   4 NiN还有一个设计与AlexNet显著不同：
        NiN去掉了AlexNet最后的3个全连接层，取而代之地，NiN使用了输出通道数等于标签类别数的NiN块，
        然后使用全局平均池化层对每个通道中所有元素求平均并直接用于分类
        这里的全局平均池化层即窗口形状等于输入空间维形状的平均池化层
        NiN的这个设计的好处是可以显著减小模型参数尺寸，从而缓解过拟合。
        然而，该设计有时会造成获得有效模型的训练时间的增加
"""
net = tf.keras.models.Sequential()
net.add(nin_block(96, kernel_size=11, strides=4, padding='valid'))
net.add(tf.keras.layers.MaxPool2D(pool_size=3, strides=2))
net.add(nin_block(256, kernel_size=5, strides=1, padding='same'))
net.add(tf.keras.layers.MaxPool2D(pool_size=3, strides=2))
net.add(nin_block(384, kernel_size=3, strides=1, padding='same'))
net.add(tf.keras.layers.MaxPool2D(pool_size=3, strides=2))

net.add(tf.keras.layers.Dropout(0.5))
# 添加输出通道数等于标签类别数的NiN块
net.add(nin_block(10, kernel_size=3, strides=1, padding='same'))
# 全局平均池化层对每个通道中所有元素求平均并直接用于分类
net.add(tf.keras.layers.GlobalAveragePooling2D())
net.add(tf.keras.layers.Flatten())


# 构造一个高和宽均为224的单通道数据样本来观察每一层的输出形状
X = tf.random.uniform((1,224,224,1))
for blk in net.layers:
    X = blk(X)
    print(blk.name, 'output shape:\t', X.shape)

"""
X: (1,224,224,1)
sequential_1 output shape:	 (1, 54, 54, 96)    ---->(1,224,224,1)输入数据经过卷积块后：(1, 54, 54, 96)
max_pooling2d output shape:	 (1, 26, 26, 96)    ---->(1, 54, 54, 96)经过步幅为2的池化后：(1, 26, 26, 96)
sequential_2 output shape:	 (1, 26, 26, 256)
max_pooling2d_1 output shape:	 (1, 12, 12, 256)
sequential_3 output shape:	 (1, 12, 12, 384)
max_pooling2d_2 output shape:	 (1, 5, 5, 384)
dropout output shape:	 (1, 5, 5, 384)
sequential_4 output shape:	 (1, 5, 5, 10)
global_average_pooling2d output shape:	 (1, 10)
flatten output shape:	 (1, 10)
"""

print("----------------获取数据和训练模型-----------------")
import numpy as np

class DataLoader():
    def __init__(self):
        fashion_mnist = tf.keras.datasets.fashion_mnist
        (self.train_images, self.train_labels), (self.test_images, self.test_labels) = fashion_mnist.load_data()
        self.train_images = np.expand_dims(self.train_images.astype(np.float32)/255.0,axis=-1)
        self.test_images = np.expand_dims(self.test_images.astype(np.float32)/255.0,axis=-1)
        self.train_labels = self.train_labels.astype(np.int32)
        self.test_labels = self.test_labels.astype(np.int32)
        self.num_train, self.num_test = self.train_images.shape[0], self.test_images.shape[0]

    def get_batch_train(self, batch_size):
        index = np.random.randint(0, np.shape(self.train_images)[0], batch_size)
        #need to resize images to (224,224)
        resized_images = tf.image.resize_with_pad(self.train_images[index],224,224,)
        return resized_images.numpy(), self.train_labels[index]

    def get_batch_test(self, batch_size):
        index = np.random.randint(0, np.shape(self.test_images)[0], batch_size)
        #need to resize images to (224,224)
        resized_images = tf.image.resize_with_pad(self.test_images[index],224,224,)
        return resized_images.numpy(), self.test_labels[index]

batch_size = 128
dataLoader = DataLoader()
x_batch, y_batch = dataLoader.get_batch_train(batch_size)
print("x_batch shape:",x_batch.shape,"y_batch shape:", y_batch.shape)


def train_nin():
    net.load_weights("5.8_nin_weights.h5")
    epoch = 5
    num_iter = dataLoader.num_train//batch_size
    for e in range(epoch):
        for n in range(num_iter):
            x_batch, y_batch = dataLoader.get_batch_train(batch_size)
            net.fit(x_batch, y_batch)
            if n%20 == 0:
                net.save_weights("5.8_nin_weights.h5")

# optimizer = tf.keras.optimizers.SGD(learning_rate=0.06, momentum=0.3, nesterov=False)
optimizer = tf.keras.optimizers.Adam(lr=1e-7)
net.compile(optimizer=optimizer,
              loss='sparse_categorical_crossentropy',
              metrics=['accuracy'])

x_batch, y_batch = dataLoader.get_batch_train(batch_size)
net.fit(x_batch, y_batch)
# train_nin()

# 将训练好的参数读入，然后取测试数据计算测试准确率
net.load_weights("5.8_nin_weights.h5")

x_test, y_test = dataLoader.get_batch_test(2000)
net.evaluate(x_test, y_test, verbose=2)








