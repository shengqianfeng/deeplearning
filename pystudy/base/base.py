# Python 3.X对于浮点数默认的是提供17位数字的精度。
print(1.2 - 1.0)    # 0.19999999999999996

# 将精度高的浮点数转换成精度低的浮点数。

"""
round函数：
    round()如果只有一个数作为参数，不指定位数的时候，返回的是一个整数，而且是最靠近的整数
"""
print(round(1.2))   # 1

# 当出现.5的时候，两边的距离都一样，round()取靠近的偶数
print(round(1.5))   # 2
print(round(2.5))   # 1

# 当指定取舍的小数点位数的时候，一般情况也是使用四舍五入的规则
# 但是碰到.5的这样情况，如果要取舍的位数前的小数是奇数，则向下取舍，如果偶数则向上取舍
print(round(2.636, 2))  # 2.64
print(round(2.645, 2))  # 2.65

print(round(2.625, 2))  # 2.62  奇怪的是并没有向上取舍 真是醉了

print(round(2.635, 2))  # 2.63

print(round(2.615, 2))  # 2.62
"""
round结论：网上说这个尽量避免使用在精度要求高的时候
"""

# dir = r"\this\is\my\dos\dir" "\\"   # \this\is\my\dos\dir\
# dir = r"\this\is\my\dos\dir\ "[:-1]     # \this\is\my\dos\dir\
dir = "\\this\\is\\my\\dos\\dir\\"   # \this\is\my\dos\dir\
print(dir)

import os
print(os.environ['PATH'])



print("" == None)   # False
print("" is None)   # False
print(not "")   # True


print("abc d".split( ));       # 以空格为分隔符，包含 \n  ['abc', 'd']

print(pow(0.5,10))
print(pow(0.75,8)*pow(0.25,2)*45)

import math
print((math.ceil(10 / 2))) # 5
print((math.ceil(9 / 2))) # 5
print((math.ceil(7 / 2))) # 4
a = [1,2,3,4,5,6,7,8,9]
print(a[:(math.ceil(len(a) / 2))])  # [1, 2, 3, 4, 5]
a = [1,2,3,4,5,6,7,8,10]
print(a[:(math.ceil(len(a) / 2))])  # [1, 2, 3, 4, 5]
print(1/2)
print(math.ceil(5027.5))

import numpy as np
x = np.array([[0.1, 0.8, 0.1],# 1
              [0.3, 0.1, 0.6],# 2
              [0.2, 0.5, 0.3],# 1
              [0.8, 0.1, 0.1]]) # 0
# 沿着第1维方向（以第1维为轴）找到值最大的元素的索引
y = np.argmax(x, axis=1)
print(y) # [1 2 1 0]



"""
在NumPy数组之间使用比较运算符（==）生成由 True/False构成的布尔
型数组，并计算True的个数
"""
y = np.array([1, 2, 1, 0])
t = np.array([1, 2, 0, 0])
print(y==t) # [ True  True False  True]
print(np.sum(y==t)) # 计算True的个数为3

a = [1,2,3,4]
print(np.array(a).size) # 4
b = np.array(a).reshape(1, np.array(a).size)
batch_size = b.shape[0]
print(batch_size)   # 1



print("-----meshgrid------")
# x0 = np.arange(-2, 2.5, 0.25)
# x1 = np.arange(-2, 2.5, 0.25)
# X, Y = np.meshgrid(x0, x1)
# X = X.flatten()
# Y = Y.flatten()
# print(X)
print("---------------------------------------")
# print(Y)
print("---------------X,Y-------------------------")
# XY = np.array([X, Y])
# print(XY)

print("=========mask的作用：小于等于0存储为true，大于0存储为false=============")
x = np.array( [[1.0, -0.5], [-2.0, 3.0]] )
print(x)
mask = (x <= 0)
"""
[[False  True]
 [ True False]]
"""
print(mask)
"""
[[ 1.  -0.5]
 [-2.   3. ]]
"""
out = x.copy()
print(out)
# 矩阵转换： 大于0的原样不变，小于0变为0
out[mask] = 0
"""
[[1. 0.]
 [0. 3.]]
"""
print(out)
print("------reshape-----------------")
x = np.array([0.1, 0.8, 0.1])
print(x.ndim)
x = x.reshape(1,x.size)
print(x.shape)  # (1, 3)
print(x.shape[0])   # 1

print("-----------------多维数组----------------")
x = np.random.rand(10, 1, 28, 28) # 生成四维数组
print(x.shape)  # (10, 1, 28, 28)
# 访问第一个 元素
# print( x[0].shape ) # (1, 28, 28)
# 访问第二个元素
# print( x[1].shape) # (1, 28, 28)

# 要访问第1个数据的第1个通道的空间数据
# print(x[0, 0])
# print(x[0][0])

print("===============")
a = list( range(0, 10) )
for i in range(0, 10, 3):
    a_batch = a[i:i+3]
    print(a_batch)
print([i for i in range(2)])


print("---------------------")
import datetime
import time

ltime = time.localtime(1581053368)
timeStr = time.strftime("%Y-%m-%d %H:%M:%S", ltime)
print(timeStr)
print("-------------------")
a=[]
b= [1,2,3]
c= [4,5,6]
a.extend(b)
a.extend(c)
print(a)
print("---------------")
date_str = '2020-02-08 12:51:42'
date = date_str[0:10]
print(date)
today = datetime.datetime.now().strftime('%Y-%m-%d')
print(today)
print(today==date)
print("-----------------")
date_str = '02-08 10:27'
date = date_str[0:5]
print(date)
today = datetime.datetime.now().strftime('%m-%d')
print(today)
print(today==date)
print("-----------")
ltime = time.localtime(1581229981)
timeStr = time.strftime("%Y-%m-%d %H:%M:%S", ltime)
print(timeStr)
print("---------------------")
url='/n1/2020/0125/c404214-31562146.html'
date_list = url[4:13].split('/')
print(date_list)
print("-----------")
print(list(range(10)))
print("------------------------")
print(list(range(0,200,20)))
print("--------------------")
ltime = time.localtime(1584255260000/1000)
timeStr = time.strftime("%Y-%m-%d %H:%M:%S", ltime)
print(timeStr)
print("---------------")
id_set=set()
print(id_set.add("1"))
print(id_set.add("2"))
print(id_set.add("3"))
print(id_set.__contains__("1"))
print(id_set.__contains__("4"))
print("---------------------------")
print(list(range(100, 200)))
print(list(range(0,3000,10)))
print("----------------------")
# ltime = time.localtime('2020-03-16T01:22:12.464Z')
# timeStr = time.strftime("%Y-%m-%d %H:%M:%S", ltime)
# print(timeStr)
# print(0%100==0)
print('2020-03-16T02:27:57-04:00'[0:10])
print("-----------")

from datetime import datetime, timedelta
now_time = datetime.now()
utc_time = now_time - timedelta(hours=8)  # UTC只是比北京时间提前了8个小时
utc_time = utc_time.strftime("%Y-%m-%dT%H:%M:%SZ")  # 转换成Aliyun要求的传参格式...
print(utc_time)
print("=======================")
from tzlocal import get_localzone
# 2020-03-19T04:06:22.473Z
time_a = '2020-03-17T07:38:01.474Z'
utc_time = datetime.strptime(time_a,'%Y-%m-%dT%H:%M:%S.%fZ')
local_time = utc_time+timedelta(hours=8)
utc_string = local_time.strftime('%Y-%m-%d %H:%M:%S')
print(utc_string)

time_b = '2020-03-17T04:32:55Z'
utc_time = datetime.strptime(time_b,'%Y-%m-%dT%H:%M:%SZ')
local_time = utc_time-timedelta(hours=4)
utc_string = local_time.strftime('%Y-%m-%d %H:%M:%S')
print(utc_string)
print("===========================")
time_b = '2020-03-17T04:32:55-04:00'
print(time_b[0:19])
print(time_b[20:])
utc_time = datetime.strptime(time_b[0:19]+"Z",'%Y-%m-%dT%H:%M:%SZ')
local_time = utc_time-timedelta(hours=int(time_b[20:][0:2]))
utc_string = local_time.strftime('%Y-%m-%d %H:%M:%S')
print(utc_string)

