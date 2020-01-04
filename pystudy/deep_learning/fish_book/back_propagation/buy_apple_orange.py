# coding: utf-8
from pystudy.nn_study.back_propagation.layer_naive import *

"""
加法与乘法层结合实现：正向传播与反向传播
@author jeffsmile
"""

apple = 100 # 苹果单价
apple_num = 2   # 苹果购买个数
orange = 150    # 橘子单价
orange_num = 3  # 橘子购买个数
tax = 1.1   # 税费

# layer 苹果乘法层
mul_apple_layer = MulLayer()
# 橘子乘法层
mul_orange_layer = MulLayer()
# 苹果橘子加法层
add_apple_orange_layer = AddLayer()
# 税费层
mul_tax_layer = MulLayer()

# forward 前向传播计算苹果价格
apple_price = mul_apple_layer.forward(apple, apple_num)  # (1)
# 前向传播计算橘子价格
orange_price = mul_orange_layer.forward(orange, orange_num)  # (2)

#前向计算苹果和橘子的总价
all_price = add_apple_orange_layer.forward(apple_price, orange_price)  # (3)
# 前向计算苹果橘子总价加上税费
price = mul_tax_layer.forward(all_price, tax)  # (4)

# backward 后向传播的计算
dprice = 1
# 后向计算总价的改变对税费和苹果橘子总价各自的影响（乘法层）----总价值=苹果橘子总价*税费
dall_price, dtax = mul_tax_layer.backward(dprice)  # (4)
# 后向计算苹果橘子总价对苹果和橘子单价各自的影响（加法层）----苹果橘子总价=苹果总价+橘子总价
dapple_price, dorange_price = add_apple_orange_layer.backward(dall_price)  # (3)
# 乘法
# 橘子总价对橘子单价和橘子数量的影响（乘法层）---橘子总价=橘子单价*橘子数量
dorange, dorange_num = mul_orange_layer.backward(dorange_price)  # (2)
# 苹果总价对苹果单价和苹果数量的影响（乘法层）---苹果总价=苹果单价*苹果数量
dapple, dapple_num = mul_apple_layer.backward(dapple_price)  # (1)

print("price:", int(price)) # 总价值715
print("dApple:", dapple)    # 苹果单价的导数2.2
print("dApple_num:", int(dapple_num))   # 苹果数量的导数110
print("dOrange:", dorange)  # 橘子单价的导数3.3
print("dOrange_num:", int(dorange_num)) # 橘子数量的导数165
print("dTax:", dtax)    # 税费的导数650
