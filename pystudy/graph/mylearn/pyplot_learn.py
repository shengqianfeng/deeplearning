
"""
绘图库的学习

"""
#导入包
import matplotlib.pyplot as plt
import numpy as np


# 绘制简单的曲线
# plt.plot([1,3,5],[4,8,10])
# plt.show()


# 集合numpy来绘制图
# x = np.linspace(-np.pi,np.pi,100)   # x轴的定义域-3.14~3.14，中间间隔100个元素
# plt.plot(x,np.sin(x))
# # 显示所画的图
# plt.show()


# 绘制多条曲线
# x = np.linspace(-np.pi*2,np.pi*2,100) # 定义域-2pi~2pi
# plt.figure(1,dpi=50)    # 创建图表1 精度50，精度越高，图片也越清晰，大小就越大
# for i in range(1,5):
#     plt.plot(x,np.sin(x/i))
# plt.show()

# 绘制直方图
# plt.figure(1,dpi=50)    # 创建图表1 精度50，精度越高，图片也越清晰，大小就越大
# data=[1,1,1,2,2,2,3,3,4,5,5,6,4]
# plt.hist(data)  # 只要传入数据直方图就会统计数据出现的次数
# plt.show()


# 绘制散点图
# x=np.arange(1,10)    # 生成1到9的连续数值赋值给x
# y=x
# fig=plt.figure()    # 创建图表
# plt.scatter(x,y,c='r',marker='o')    #绘制图表 c='r'表示散点图颜色为红色，maker表示指定的散点为原型
# plt.show()


