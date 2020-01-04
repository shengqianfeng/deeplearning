# coding: utf-8

"""
反向传播-乘法层
"""
class MulLayer:
    def __init__(self):
        self.x = None
        self.y = None
    # 前向传播 计算
    def forward(self, x, y):
        self.x = x
        self.y = y                
        out = x * y

        return out

    # 反向传播
    def backward(self, dout):
        dx = dout * self.y
        dy = dout * self.x

        return dx, dy


class AddLayer:
    def __init__(self):
        pass

    def forward(self, x, y):
        out = x + y

        return out

    def backward(self, dout):
        dx = dout * 1
        dy = dout * 1

        return dx, dy
