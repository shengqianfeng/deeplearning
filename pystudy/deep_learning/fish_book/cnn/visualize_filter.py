# coding: utf-8
import numpy as np
import matplotlib.pyplot as plt
from pystudy.nn_study.cnn.simple_convNet import SimpleConvNet

def filter_show(filters, nx=8, margin=3, scale=10):
    """
    c.f. https://gist.github.com/aidiary/07d530d5e08011832b12#file-draw_weight-py
    """
    FN, C, FH, FW = filters.shape
    ny = int(np.ceil(FN / nx))

    fig = plt.figure()
    # 调整边距和子图的间距  hspace为子图之间的空间保留的宽度，平均轴宽的一部分  wspace为子图之间的空间保留的高度，平均轴高度的一部分
    fig.subplots_adjust(left=0, right=1, bottom=0, top=1, hspace=0.05, wspace=0.05)

    for i in range(FN):
        # add_subplot参数1：子图总行数   参数2：子图总列数  参数3：子图位置
        ax = fig.add_subplot(ny, nx, i+1, xticks=[], yticks=[])
        # imshow 参数1：要绘制的图像或数组 cmap：颜色图谱
        ax.imshow(filters[i, 0], cmap=plt.cm.gray_r, interpolation='nearest')
    plt.show()


network = SimpleConvNet()
# 随机进行初始化后的权重
filter_show(network.params['W1'])

# 学习后的权重
network.load_params("params.pkl")
filter_show(network.params['W1'])