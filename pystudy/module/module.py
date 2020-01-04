#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
模块(packageA.packageB.xxx.py)：
己创建模块时要注意命名，不能和Python自带的模块名称冲突。例如，系统自带了sys模块，自己的模块就不可命名为sys.py，否则将无法导入系统自带的sys模块
"""

__author__ = 'jeff Sheng'

import sys

def test():
    # sys模块有一个argv变量，用list存储了命令行的所有参数。argv至少有一个元素，因为第一个参数永远是该.py文件的名称，运行python3 hello.py获得的sys.argv就是['hello.py']
    # 运行python3 hello.py Michael获得的sys.argv就是['hello.py', 'Michael]
    args = sys.argv
    if len(args) == 1:
        print('Hello, world!')
    elif len(args)==2:
        print('Hello, %s!' % args[1])
    else:
        print('Too many arguments!')


if __name__=='__main__':
    test()


"""
访问控制：
1 正常的函数和变量名是公开的（public），可以被直接引用，比如：abc，x123，PI等
2 类似__xxx__这样的变量是特殊变量，可以被直接引用，但是有特殊用途，比如上面的__author__，__name__就是特殊变量，hello模块定义的文档注释也可以用特殊变量__doc__访问，我们自己的变量一般不要用这种变量名；
3 类似_xxx和__xxx这样的函数或变量就是非公开的（private），不应该被直接引用，比如_abc，__abc等
4 private函数和变量“不应该”被直接引用，而不是“不能”被直接引用，是因为Python并没有一种方法可以完全限制访问private函数或变量，但是，从编程习惯上不应该引用private函数或变量
"""


def _private_1(name):
    return 'Hello, %s' % name


def _private_2(name):
    return 'Hi, %s' % name


def greeting(name):
    if len(name) > 3:
        return _private_1(name)
    else:
        return _private_2(name)


"""
我们在模块里公开greeting()函数，而把内部逻辑用private函数隐藏起来了，这样，调用greeting()函数不用关心内部的private函数细节，这也是一种非常有用的代码封装和抽象的方法
即：外部不需要引用的函数全部定义成private，只有外部需要引用的函数才定义为public。
"""

import pystudy.oop.obj as _obj_

t = _obj_.Teacher("张三")

############属性####################
print(getattr(t, "name"))    # 张三  获取实例的属性值
# n = getattr(t, "name")   根本没name这个函数，当然会报错了
# n() # TypeError: 'str' object is not callable
print("我是:", t.name) # 我是: 张三
# print("我的外号:", t.__other_name)   # 外号不能让你知道！哈哈  异常： AttributeError: 'Teacher' object has no attribute '__other_name'
print("我的书名:",t.__book_name__)  # 我的书名: 张三  可以看到区别只是属性只是后边多俩下划线而已就可以被访问

############函数####################
t.jiangke()  # 我正在讲课.....   直接调用实例的函数
jiangke = getattr(t, "jiangke")  # 获取函数值
jiangke()   # # 我正在讲课..... 直接调用函数
getattr(t, "jiangke")()     # 我正在讲课.....  尼玛这也可以

