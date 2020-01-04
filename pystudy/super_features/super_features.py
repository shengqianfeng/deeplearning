"""
python高级特性
"""

# 取一个list或tuple的部分元素
L = ['Michael', 'Sarah', 'Tracy', 'Bob', 'Jack']

# 取前N个元素，也就是索引为0-(N-1)的元素，可以用循环
r = []
n = 3
for i in range(n):
    r.append(L[i])
print(r)    # ['Michael', 'Sarah', 'Tracy']

# 用循环十分繁琐，因此，Python提供了切片（Slice）操作符，能大大简化这种操作
"""
切片
"""
# 取前3个元素，用一行代码就可以完成切片
print(L[0:3])   # ['Michael', 'Sarah', 'Tracy']
# L[0:3]表示，从索引0开始取，直到索引3为止，但不包括索引3。即索引0，1，2，正好是3个元素
# 如果第一个索引是0，还可以省略
print(L[:3])   # ['Michael', 'Sarah', 'Tracy']

# 也可以从索引1开始，取出2个元素出来
print(L[1:3])   # ['Sarah', 'Tracy']

# 既然Python支持L[-1]取倒数第一个元素，那么它同样支持倒数切片
print(L[-2:])   # ['Bob', 'Jack']
print(L[-2:-1])    # ['Bob']    记住倒数第一个元素的索引是-1。


# 打印0-99数列
L = list(range(100))
# print(L)
# 可以通过切片轻松取出某一段数列。比如前10个数
print( L[:10]) # [0, 1, 2, 3, 4, 5, 6, 7, 8, 9]

# 后10个数
print(L[-10:])  # [90, 91, 92, 93, 94, 95, 96, 97, 98, 99]

#  前11-20个数
print(L[10:20])  # [10, 11, 12, 13, 14, 15, 16, 17, 18, 19]

# 前10个数，每两个取一个
print(L[:10:2]) # [0, 2, 4, 6, 8]

#  所有数，每5个取一个
print(L[::5])  # [0, 5, 10, 15, 20, 25, 30, 35, 40, 45, 50, 55, 60, 65, 70, 75, 80, 85, 90, 95]

#  甚至什么都不写，只写[:]就可以原样复制一个list
print(L[:])
# tuple也是一种list，唯一区别是tuple不可变。因此，tuple也可以用切片操作，只是操作的结果仍是tuple：
print((0, 1, 2, 3, 4, 5)[:3])   # (0, 1, 2)

# 字符串'xxx'也可以看成是一种list，每个元素就是一个字符。因此，字符串也可以用切片操作，只是操作结果仍是字符串
print('ABCDEFG'[:3])  # ABC
print('ABCDEFG'[::2])   # ACEG

# Python没有针对字符串的截取函数，只需要切片一个操作就可以完成，非常简单



"""
迭代：
    如果给定一个list或tuple，我们可以通过for循环来遍历这个list或tuple，这种遍历我们称为迭代（Iteration）
1 在Python中，迭代是通过for ... in来完成的
2 Python的for循环不仅可以用在list或tuple上，还可以作用在其他可迭代对象上
3 ist这种数据类型虽然有下标，但很多其他数据类型是没有下标的，但是，只要是可迭代对象，
无论有无下标，都可以迭代，比如dict就可以迭代
"""
d = {'a': 1, 'b': 2, 'c': 3}
for key in d:
    print(key)  # 因为dict的存储不是按照list的方式顺序排列，所以，迭代出的结果顺序很可能不一样

"""
默认情况下，dict迭代的是key。如果要迭代value，可以用
    for value in d.values()，
如果要同时迭代key和value，可以用
    for k, v in d.items()
"""
# 由于字符串也是可迭代对象，因此，也可以作用于for循环：
for ch in 'ABC':
    print(ch)

"""
当我们使用for循环时，只要作用于一个可迭代对象，for循环就可以正常运行，
而我们不太关心该对象究竟是list还是其他数据类型

如何判断一个对象是可迭代对象呢？---方法是通过collections模块的Iterable类型判断
"""
from collections import Iterable
print(isinstance('abc', Iterable))  # str是否可迭代 True
print(isinstance([1,2,3], Iterable))  #  list是否可迭代 True
print(isinstance(123, Iterable))    # 整数是否可迭代 False

"""
如果要对list实现类似Java那样的下标循环怎么办？
Python内置的enumerate函数可以把一个list变成索引-元素对，这样就可以在for循环中同时迭代索引和元素本身：
"""
for i, value in enumerate(['A', 'B', 'C']):
     print(i, value)


# 上面的for循环里，同时引用了两个变量，在Python里是很常见的
for x, y in [(1, 1), (2, 4), (3, 9)]:
    print(x, y)


"""
列表生成式
    列表生成式即List Comprehensions，是Python内置的非常简单却强大的可以用来创建list的生成式。
"""
# 要生成list [1, 2, 3, 4, 5, 6, 7, 8, 9, 10]可以用list(range(1, 11))：
print(list(range(1, 11)))   # [1, 2, 3, 4, 5, 6, 7, 8, 9, 10]


# 如果要生成[1x1, 2x2, 3x3, ..., 10x10]怎么做
# 方法一是循环
L = []
for x in range(1, 11):
    L.append(x * x)
print(L)    # [1, 4, 9, 16, 25, 36, 49, 64, 81, 100]

# 列表生成式则可以用一行语句代替循环生成上面的list
print([x * x for x in range(1, 11)])    # [1, 4, 9, 16, 25, 36, 49, 64, 81, 100]
# 写列表生成式时，把要生成的元素x * x放到前面，后面跟for循环，就可以把list创建出来

# for循环后面还可以加上if判断，这样我们就可以筛选出仅偶数的平方
print([x * x for x in range(1, 11) if x % 2 == 0])  # [4, 16, 36, 64, 100]

#  还可以使用两层循环，可以生成全排列
print([m + n for m in 'ABC' for n in 'XYZ'])    # ['AX', 'AY', 'AZ', 'BX', 'BY', 'BZ', 'CX', 'CY', 'CZ']

# 出当前目录下的所有文件和目录名，可以通过一行代码实现
import os  # 导入os模块，模块的概念后面讲到
print([d for d in os.listdir('.')]) #  # os.listdir可以列出文件和目录
# ['dict_set.py', 'func.py', 'if_else.py', 'list_tuple.py', 'loop_console.py', 'str_encode.py', 'super_features.py', '__init__.py']

# for循环其实可以同时使用两个甚至多个变量，比如dict的items()可以同时迭代key和value
d = {'x': 'A', 'y': 'B', 'z': 'C' }
for k, v in d.items():
    print(k, '=', v)

d = {'x': 'A', 'y': 'B', 'z': 'C' }
print([k + '=' + v for k, v in d.items()])  # ['x=A', 'y=B', 'z=C']

# 把一个list中所有的字符串变成小写
L = ['Hello', 'World', 'IBM', 'Apple']
print([s.lower() for s in L])   # ['hello', 'world', 'ibm', 'apple']


"""
生成器：generator
    如果列表元素可以按照某种算法推算出来，那我们是否可以在循环的过程中不断推算出后续的元素呢？这样就不必创建完整的list，
从而节省大量的空间。
在Python中，这种一边循环一边计算的机制，称为生成器generator：
"""
# 要创建一个generator，有很多种方法。
# 第一种方法很简单，只要把一个列表生成式的[]改成()，就创建了一个generator
L = [x * x for x in range(10)]
print(L)    # [0, 1, 4, 9, 16, 25, 36, 49, 64, 81]
g = (x * x for x in range(10))
print(g)    # <generator object <genexpr> at 0x0000014E8EEB9748>

"""
列表推导式与生成器：
    创建L和g的区别仅在于最外层的[]和()，L是一个list，而g是一个generator。
我们可以直接打印出list的每一个元素，但我们怎么打印出generator的每一个元素呢

next()函数：
    如果要一个一个打印出来，可以通过next()函数获得generator的下一个返回值
"""
print(next(g))  # 0
print(next(g))  # 1
print(next(g))  # 4
print(next(g))  # 9
# generator保存的是算法，每次调用next(g)，就计算出g的下一个元素的值，
# 直到计算到最后一个元素，没有更多的元素时，抛出StopIteration的错误

# 面这种不断调用next(g)实在是太变态了，正确的方法是使用for循环，因为generator也是可迭代对象并且不需要关心StopIteration的错误

g = (x * x for x in range(10))
print("通过for循环迭代生成器........start")
for n in g:
    print(n)
print("通过for循环迭代生成器........end")


"""
generate函数：（普通函数+yeild--->转化为generate函数）
    如果一个函数定义中包含yield关键字，那么这个函数就不再是一个普通函数，而是一个generator
"""


# 菲波那切数列生成器
def fib(max):
    n, a, b = 0, 0, 1
    while n < max:
        yield b
        a, b = b, a + b
        n = n + 1
    return 'done'


f = fib(6)
print(f)    # <generator object fib at 0x104feaaa0>
print("----------------")
"""
以下输出：
1
1
2
3
5
8
处理步骤：
    遇到yeild b就返回b的值，然后print打印出来，打印完毕继续generate函数中yeild后边的处理
"""
for n in fib(6):
    print(n)
print("----------------")
# 用for循环调用generator时，发现拿不到generator的return语句的返回值。
# 。如果想要拿到返回值，必须捕获StopIteration错误，返回值包含在StopIteration的value中
while True:
    try:
        x = next(f)
        print('f:', x)
    except StopIteration as e:
        print('Generator return value:', e.value)
        break
print("+++++++++++++++++++++")



"""
迭代器
    可以直接作用于for循环的数据类型有以下几种
一类是集合数据类型，如list、tuple、dict、set、str等；
一类是generator，包括生成器和带yield的generator function。

这些可以直接作用于for循环的对象统称为可迭代对象：Iterable。
"""

print("----------")
# def triangles():
#     for n in range(6):
#         results = []
#         for m in range(n+1):
#             results.append(m+1)
#         print(results)
# triangles()