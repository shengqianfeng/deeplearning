
# 要获得函数调用结果，我们可以把结果赋值给变量
x = abs(-10)
print(x)

# 变量f现在已经指向了abs函数本身。直接调用abs()函数和调用变量f()完全相同
f = abs
print(f(-10))

# 函数名其实就是指向函数的变量！对于abs()这个函数，完全可以把函数名abs看成变量，它指向一个可以计算绝对值的函数
"""
把abs指向10后，就无法通过abs(-10)调用该函数了！因为abs这个变量已经不指向求绝对值函数而是指向一个整数10！
"""
# abs = 10
# print(abs(-10))     # TypeError: 'int' object is not callable

"""
高阶函数：
    既然变量可以指向函数，函数的参数能接收变量，那么一个函数就可以接收另一个函数作为参数，这种函数就称之为高阶函数。
"""


def add(x, y, f):
    return f(x) + f(y)


x = -5
y = 6
f = abs
print(add(x,y,abs)) # 11
# 编写高阶函数，就是让函数的参数能够接收别的函数



"""
map/reduce
Python内建了map()和reduce()函数。
map()函数接收两个参数，一个是函数，一个是Iterable，map将传入的函数依次作用到序列的每个元素，并把结果作为新的Iterator返回
比如我们有一个函数f(x)=x2，要把这个函数作用在一个list [1, 2, 3, 4, 5, 6, 7, 8, 9]上

"""

def f(x):
    return x * x


r = map(f, [1, 2, 3, 4, 5, 6, 7, 8, 9])
# 结果r是一个Iterator，Iterator是惰性序列，因此通过list()函数让它把整个序列都计算出来并返回一个list
print(list(r))  # [1, 4, 9, 16, 25, 36, 49, 64, 81]
# 把这个list所有数字转为字符串
print(list(map(str, [1, 2, 3, 4, 5, 6, 7, 8, 9])))  # ['1', '2', '3', '4', '5', '6', '7', '8', '9']


"""
reduce把一个函数作用在一个序列[x1, x2, x3, ...]上，这个函数必须接收两个参数，reduce把结果继续和序列的下一个元素做累积计算
reduce(f, [x1, x2, x3, x4]) = f(f(f(x1, x2), x3), x4)
"""

from functools import reduce


def add(x, y):
    return x + y


num = reduce(add, [1, 3, 5, 7, 9])
print(num)  # 25


# 把序列[1, 3, 5, 7, 9]变换成整数13579
def fn(x, y):
    return x * 10 + y


num = reduce(fn,[1,3,5,7,9])
print(num)  # 13579


def char2num(s):
    digits = {'0': 0, '1': 1, '2': 2, '3': 3, '4': 4, '5': 5, '6': 6, '7': 7, '8': 8, '9': 9}
    return digits[s]


num = reduce(fn, map(char2num, '13579'))
print(num)  # 13579

# 整理成一个str2int的函数
DIGITS = {'0': 0, '1': 1, '2': 2, '3': 3, '4': 4, '5': 5, '6': 6, '7': 7, '8': 8, '9': 9}
def str2int(s):
    def fn(x, y):
        return x * 10 + y
    def char2num(s):
        return DIGITS[s]
    return reduce(fn, map(char2num, s))


# 用lambda函数进一步简化
def char2num(s):
    return DIGITS[s]

def str2int(s):
    return reduce(lambda x, y: x * 10 + y, map(char2num, s))



"""
1 Python内建的filter()函数用于过滤序列
2 和map()类似，filter()也接收一个函数和一个序列
3 和map()不同的是，filter()把传入的函数依次作用于每个元素，然后根据返回值是True还是False决定保留还是丢弃该元素
4 filter()函数返回的是一个Iterator，也就是一个惰性序列，所以要强迫filter()完成计算结果，需要用list()函数获得所有结果并返回list
"""


# 在一个list中，删掉偶数，只保留奇数
def is_odd(n):
    return n % 2 == 1


r = list(filter(is_odd, [1, 2, 4, 5, 6, 9, 10, 15]))
print(r)    # [1, 5, 9, 15]


# 把一个序列中的空字符串删掉
def not_empty(s):
    return s and s.strip()


print(list(filter(not_empty, ['A', '', 'B', None, 'C', '  '])))   # ['A', 'B', 'C']

"""
sort函数
"""
# Python内置的sorted()函数就可以对list进行排序
print(sorted([36, 5, -12, 9, -21]))     # [-21, -12, 5, 9, 36]

"""
sorted()函数也是一个高阶函数，它还可以接收一个key函数来实现自定义的排序
key指定的函数将作用于list的每一个元素上，并根据key函数返回的结果进行排序
"""
print(sorted([36, 5, -12, 9, -21], key=abs))    # [5, 9, -12, -21, 36]

# 字符串排序 --->默认情况下，对字符串排序，是按照ASCII的大小比较的，由于'Z' < 'a'，结果，大写字母Z会排在小写字母a的前面。
print(sorted(['bob', 'about', 'Zoo', 'Credit']))    # ['Credit', 'Zoo', 'about', 'bob']

"""
我们提出排序应该忽略大小写，按照字母序排序。要实现这个算法，不必对现有代码大加改动，
只要我们能用一个key函数把字符串映射为忽略大小写排序即可。忽略大小写来比较两个字符串，
实际上就是先把字符串都变成大写（或者都变成小写），再比较
"""
print(sorted(['bob', 'about', 'Zoo', 'Credit'], key=str.lower))     # ['about', 'bob', 'Credit', 'Zoo']
# 要进行反向排序，不必改动key函数，可以传入第三个参数reverse=True
print(sorted(['bob', 'about', 'Zoo', 'Credit'], key=str.lower, reverse=True))   # ['Zoo', 'Credit', 'bob', 'about']


"""
高阶函数除了可以接受函数作为参数外，还可以把函数作为结果值返回
"""
# 来实现一个可变参数的求和
def calc_sum(*args):
    ax = 0
    for n in args:
        ax = ax + n
    return ax
"""
如果不需要立刻求和，而是在后面的代码中，根据需要再计算怎么办？可以不返回求和的结果，而是返回求和的函数
"""
def lazy_sum(*args):
    def sum():
        ax = 0
        for n in args:
            ax = ax + n
        return ax
    return sum

# 当我们调用lazy_sum()时，返回的并不是求和结果，而是求和函数
f = lazy_sum(1, 3, 5, 7, 9)
print(f)    # <function lazy_sum.<locals>.sum at 0x000001BBBF76D168>

# 调用函数f时，才真正计算求和的结果
print(f())  # 25

"""
在这个例子中，我们在函数lazy_sum中又定义了函数sum，并且，内部函数sum可以引用外部函数lazy_sum的参数和局部变量，
当lazy_sum返回函数sum时，相关参数和变量都保存在返回的函数中，这种称为“闭包（Closure）”的程序结构拥有极大的威力
"""
# 当我们调用lazy_sum()时，每次调用都会返回一个新的函数，即使传入相同的参数
f1 = lazy_sum(1, 3, 5, 7, 9)
f2 = lazy_sum(1, 3, 5, 7, 9)
print(f1 == f2)   # False


"""
闭包
"""
def count():
    fs = []
    for i in range(1, 4):
        def f():
             return i*i
        fs.append(f)
    return fs

f1, f2, f3 = count()

# 每次循环，都创建了一个新的函数，然后，把创建的3个函数都返回了
print(f1())     # 9
print(f2())     # 9
print(f3())     # 9
"""
全部都是9！原因就在于返回的函数引用了变量i，但它并非立刻执行。等到3个函数都返回时，它们所引用的变量i已经变成了3，因此最终结果为9
返回闭包时牢记一点：返回函数不要引用任何循环变量，或者后续会发生变化的变量。
"""


"""
lambdas匿名函数：
python的lambdas只是一种速记符号，如果您懒得定义函数的话
匿名函数：当我们在传入函数时，有些时候，不需要显式地定义函数，直接传入匿名函数更方便
         匿名函数有个限制，就是只能有一个表达式，不用写return，返回值就是该表达式的结果
         用匿名函数有个好处，因为函数没有名字，不必担心函数名冲突。
         此外，匿名函数也是一个函数对象，也可以把匿名函数赋值给一个变量，再利用变量来调用该函数
"""
# 在Python中，对匿名函数提供了有限支持。还是以map()函数为例，计算f(x)=x2时，除了定义一个f(x)的函数外，还可以直接传入匿名函数
print(list(map(lambda x: x * x, [1, 2, 3, 4, 5, 6, 7, 8, 9])))      # [1, 4, 9, 16, 25, 36, 49, 64, 81]


# 匿名函数lambda x: x * x实际上就是  关键字lambda表示匿名函数，冒号前面的x表示函数参数
def f(x):
    return x * x


f = lambda x: x * x
print(f(10))


# 同样，也可以把匿名函数作为返回值返回
def build(x, y):
    return lambda: x * x + y * y


"""
    语法：1 变量=函数对象
          2 变量()
    由于函数也是一个对象，而且函数对象可以被赋值给变量，所以，通过变量也能调用该函数
    
    函数对象有一个__name__属性，可以拿到函数的名字
    语法：
        1 函数对象.__name__
        2 变量.__name__
"""


def now():
    print('2015-3-25')


f = now
f()  # 2015-3-25

print(now.__name__)     # now

print(f.__name__)       # now

"""
装饰器
    假设我们要增强now()函数的功能，比如，在函数调用前后自动打印日志，但又不希望修改now()函数的定义，这种在代码运行期间动态增加功能的方式，称之为“装饰器”（Decorator)
    
    其实就是在目标函数上包装了一层用于打印日志等操作，完毕之后执行函数，相当于AOP策略。
"""

# decorator就是一个返回函数的高阶函数。所以，我们要定义一个能打印日志的decorator，可以定义如下
def log(func):
    def wrapper(*args, **kw):
        print('call %s():' % func.__name__)
        return func(*args, **kw)
    return wrapper


# 观察上面的log，因为它是一个decorator，所以接受一个函数作为参数，并返回一个函数。我们要借助Python的@语法，把decorator置于函数的定义处
@log
def now():
    print('2015-3-25')


# 调用now()函数，不仅会运行now()函数本身，还会在运行now()函数前打印一行日志
now()   # call now():  2015-3-25

# 把@log放到now()函数的定义处，相当于执行了语句
# now = log(now)

# wrapper()函数的参数定义是(*args, **kw)，因此，wrapper()函数可以接受任意参数的调用。在wrapper()函数内，首先打印日志，再紧接着调用原始函数
print(now.__name__)     # wrapper

import functools
# 以下代码可以打印原始函数的名字 @functools.wraps(func)
def log(func):
    @functools.wraps(func)
    def wrapper(*args, **kw):
        print('call %s():' % func.__name__)
        return func(*args, **kw)
    return wrapper

@log
def now():
    print('2015-5-25')


now()
print(now.__name__)     # now


# int()函数可以把字符串转换为整数，当仅传入字符串时，int()函数默认按十进制转换
print(int('12345'))     # 12345

# 但int()函数还提供额外的base参数，默认值为10。如果传入base参数，就可以做N进制的转换
print(int('12345', base=8))    # 5349
print(int('12345', 16))    # 74565


# 假设要转换大量的二进制字符串，每次都传入int(x, base=2)非常麻烦，于是，我们想到，可以定义一个int2()的函数，默认把base=2传进去
def int2(x, base=2):
    return int(x, base)


# 我们转换二进制就非常方便了，默认参数无需传入
print(int2('1000000'))      # 64
print(int2('1010101'))     # 85

"""
偏函数总结：
    简单总结functools.partial的作用就是，把一个函数的某些参数给固定住（也就是设置默认值），返回一个新的函数，调用这个新函数会更简单
    当函数的参数个数太多，需要简化时，使用functools.partial可以创建一个新的函数，这个新函数可以固定住原函数的部分参数，从而在调用时更简单。
    # functools.partial就是帮助我们创建一个偏函数的，不需要我们自己定义int2()
"""
# 可以直接使用下面的代码创建一个新的函数int2


int2 = functools.partial(int, base=2)
print(type(int2))   # <class 'functools.partial'>
print(int2('1000000'))      # 64
print(int2('1010101'))      # 85










