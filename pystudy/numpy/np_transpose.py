
import numpy as np
x = np.arange(4).reshape((2,2))
"""
[[0 1]
 [2 3]]
"""
print(x)
"""
[[0 2]
 [1 3]]
"""
"""
矩阵转置：
    对于二维 ndarray，transpose在不指定参数时默认是矩阵转置
"""
print(x.transpose())
"""
输出无变化
[[0 1]
 [2 3]]
"""
# 表示按照原坐标轴改变序列，也就是保持不变
print(x.transpose((0,1)))
"""
输出转置了 
[[0 2]
 [1 3]]
"""
#  表示交换 ‘0轴’ 和 ‘1轴’
print(x.transpose((1,0)))