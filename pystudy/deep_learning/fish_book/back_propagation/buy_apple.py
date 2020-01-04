# coding: utf-8
from pystudy.nn_study.back_propagation.layer_naive import *

"""
乘法层的实现：正向传播与反向传播
"""
# 苹果价格100日元
apple = 100
apple_num = 2   # 购买数量
tax = 1.1   # 税费

mul_apple_layer = MulLayer()
mul_tax_layer = MulLayer()

# forward 前向传播计算付款价值函数（单价，数量）不含税费
apple_price = mul_apple_layer.forward(apple, apple_num)
# 前向传播计算总费用函数（价格，税费）
price = mul_tax_layer.forward(apple_price, tax)

# backward
dprice = 1 # 后向传播总费用导数
# 苹果价值导数，税费导数
dapple_price, dtax = mul_tax_layer.backward(dprice)
# 单价导数，数量导数
dapple, dapple_num = mul_apple_layer.backward(dapple_price)

print("price:", int(price)) # price: 220  总价220
# 如果消费税和苹果的价格增加相同的值，则消费税将对最终价格产生200倍大小的影响，苹果的价格将产生2.2倍大小的影响
print("dTax:", dtax)    # dTax: 200 消费税的导数
print("dApple:", dapple)   # dApple: 2.2 苹果单价的导数
print("dApple_num:", int(dapple_num))   # dApple_num: 110  苹果个数的导数
