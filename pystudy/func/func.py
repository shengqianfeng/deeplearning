
"""
函数
"""
# 计算圆的面积
# r1 = 12.34
# r2 = 9.08
# r3 = 73.1
# s1 = 3.14 * r1 * r1
# s2 = 3.14 * r2 * r2
# s3 = 3.14 * r3 * r3

"""
函数内置调用
"""
# 求绝对值
print(abs(-100))    # 100

# 求最大值
print(max(100, 1000))    # 1000


# 数据类型转换
print(int('123'))   # 123

print(int(12.34))   # 12

print(float('12.34'))  # 12.34

print(str(1.23))    # '1.23'

print(bool(1))  # True

print( bool(''))  # False


"""
函数别名：函数名其实就是指向一个函数对象的引用，完全可以把函数名赋给一个变量，相当于给这个函数起了一个“别名”

"""
a = abs  # 变量a指向abs函数
print(a(-1))  # 1  可以通过a调用abs函数


"""
定义函数:
  如果没有return语句，函数执行完毕后也会返回结果，只是结果为None。return None可以简写为return
"""
# 自定义一个求绝对值的my_abs函数
def my_abs(x):
    if x >= 0:
        return x
    else:
        return -x
# 99
print( my_abs(-99))

"""
空函数
    1 如果想定义一个什么事也不做的空函数，可以用pass语句
   2  pass语句什么都不做，那有什么用？实际上pass可以用来作为占位符，
    比如现在还没想好怎么写函数的代码，就可以先放一个pass，让代码能运行起来
"""
def nop():
    pass

"""
函数返回多个值
"""
# import math语句表示导入math包，并允许后续代码引用math包里的sin、cos等函数
import math
def move(x, y, step, angle=0):
    nx = x + step * math.cos(angle)
    ny = y - step * math.sin(angle)
    return nx, ny

x, y = move(100, 100, 60, math.pi / 6)
print(x,"---",y)    # 151.96152422706632 --- 70.0

# 看上去确实返回了两个值，其实这只是一种假象，Python函数返回的仍然是单一值：
#   Python的函数返回多值其实就是返回一个tuple，但写起来更方便
r = move(100, 100, 60, math.pi / 6)
print(r)    # (151.96152422706632, 70.0)





"""
递归函数
"""
# 定义阶乘
# def fact(n):
#     if n==1:
#         return 1
#     return n * fact(n - 1)
#
# print(fact(1))  # 1
# print(fact(5))  # 120

#用递归函数需要注意防止栈溢出。在计算机中，函数调用是通过栈（stack）这种数据结构实现的，
# 每当进入一个函数调用，栈就会加一层栈帧，
# 每当函数返回，栈就会减一层栈帧。由于栈的大小不是无限的，所以，递归调用的次数过多，会导致栈溢出。优化方式：尾递归
# print(fact(1000))   # RecursionError: maximum recursion depth exceeded in comparison


# zip()
# 函数用于将可迭代的对象作为参数，将对象中对应的元素打包成一个个元组，然后返回由这些元组组成的列表
# 如果各个可迭代对象的元素个数不一致，则返回的对象长度与最短的可迭代对象相同
# 利用 * 号操作符，与zip相反，进行解压。

a = [1,2,3,5]
b = [4,5,6]
zipped = zip(a,b)     # 打包为元组的列表


c= list(zipped)
print(c)   # [(1, 4), (2, 5), (3, 6)]

print(*c)   # (1, 4) (2, 5) (3, 6)

print(*zip(*c)) # (1, 2, 3) (4, 5, 6)


