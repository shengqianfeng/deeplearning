#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
py的字符串和编码
"""

print('包含中文的str')

# 函数获取单个字符的整数表示
a = ord('A')
print(a)
print(ord('中'))

# 把编码转换为对应的字符
print(chr(66))
print(chr(25991))

# 根据字符的整数编码打印对应的中文，跟前边的等价
print('\u4e2d\u6587')

# 意区分'ABC'和b'ABC'，前者是str，后者虽然内容显示得和前者一样，但bytes的每个字符都只占用一个字节。
x = b'ABC'
print(x)
"""
encode编码为bytes
"""
# 以Unicode表示的str通过encode()方法可以编码为指定的bytes
print('ABC'.encode('ascii'))  # print('ABC'.encode('ascii'))
print('中文'.encode('utf-8'))  # b'\xe4\xb8\xad\xe6\x96\x87'
# 报错！ 含有中文的str无法用ASCII编码，因为中文编码的范围超过了ASCII编码的范围，Python会报错
# print('中文'.encode('ascii'))  # b'\xe4\xb8\xad\xe6\x96\x87'
"""
encode使用总结：
1 纯英文的str可以用ASCII编码为bytes，内容是一样的
2 含有中文的str可以用UTF-8编码为bytes
3 含有中文的str无法用ASCII编码为bytes，因为中文编码的范围超过了ASCII编码的范围
"""


"""
decode解码bytes为str:
如果我们从网络或磁盘上读取了字节流，那么读到的数据就是bytes。
要把bytes变为str，就需要用decode()方法
"""
print(b'ABC'.decode('ascii'))  # ABC
print(b'\xe4\xb8\xad\xe6\x96\x87'.decode('utf-8'))  # 中文
#  bytes中包含无法解码的字节
# print(b'\xe4\xb8\xad\xff'.decode('utf-8'))

# 忽略无效字节
print(b'\xe4\xb8\xad\xff'.decode('utf-8', errors='ignore'))  # 中


# 计算str包含的字符数
print(len('ABC'))  # 3
print(len('中文'))  # 2

# 计算bytes的字节数
print(len(b'ABC'))  # 3
# 因为是汉字的utf8编码每个汉字3个字节，所以是6个
print(len(b'\xe4\xb8\xad\xe6\x96\x87'))  # 中文 6
print(len('中文'.encode('utf-8')))  # 6 先编码为bytes再计算长度

# 字符串格式化
"""
%d 整数
%f 浮点数
%s  字符串   ps: 永远起作用，它会把任何数据类型转换为字符串
%x 十六进制整数
"""
# Hello, world 只有一个占位符%后不需要加括号
print('Hello, %s' % 'world')
print('Hi, %s, you have $%d.' % ('Michael', 1000000))
print('Age: %s. Gender: %s' % (25, True))
# 转义，用%%来表示一个%
print('growth rate: %d %%' % 7)


"""
另一种格式化字符串的方法是使用字符串的format()方法，它会用传入的参数依次替换字符串内的占位符{0}、{1}
比较麻烦
"""
# Hello, 小明, 成绩提升了 17.1%
print('Hello, {0}, 成绩提升了 {1:.1f}%'.format('小明', 17.125))


"""
当str和bytes互相转换时，需要指定编码。最常用的编码是UTF-8。
Python当然也支持其他编码方式，比如把Unicode编码成GB2312：
不建议使用
"""
print('中文'.encode('gb2312'))







