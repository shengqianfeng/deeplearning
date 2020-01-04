"""
激活函数层的计算图思路实现
"""
from pystudy.common.layers import *

# Relu函数测试
x = np.array([-1,2,3,-3])
relu = Relu()
print(relu.forward(x))  # [0,2,3,0]
print(relu.backward(np.array([0,1,1,0])))   # [0 1 1 0]






