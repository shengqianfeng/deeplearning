
"""
函数的参数
"""
# 位置参数
"""
计算x2的函数:
    对于power(x)函数，参数x就是一个位置参数。当我们调用power函数时，必须传入有且仅有的一个参数x
"""
def power(x):
    return x * x

print(power(5)) # 25

"""
用来计算x^n
"""
def power(x, n=2):
    s = 1
    while n > 0:
        n = n - 1
        s = s * x
    return s

print(power(5, 2))  # 25
print(power(5, 3))  # 125

# 默认参数
# print(power(5))  # TypeError: power() missing 1 required positional argument: 'n'
# 报错解释：因为新定义的power有两个参数，而旧的只有一个参数导致报错，解决办法就是形参中的默认参数def power(x, n=2)

"""
设置默认参数时，有几点要注意
一是必选参数在前，默认参数在后，否则Python的解释器会报错
二是如何设置默认参数 当函数有多个参数时，把变化大的参数放前面，变化小的参数放后面。变化小的参数就可以作为默认参数。
"""


"""
可变参数：
  语法： *变量名
在Python函数中，还可以定义可变参数。顾名思义，可变参数就是传入的参数个数是可变的，可以是1个、2个到任意个，还可以是0个。
"""

def calc1(numbers):
    sum = 0
    for n in numbers:
        sum = sum + n * n
    return sum
# 正常调用的时候，需要先组装出一个list或tuple
print(calc1([1, 2, 3]))  # 14
print(calc1((1, 3, 5, 7)))  # 84

"""
# 如果利用可变参数，调用函数的方式可以简化成这样:
#               print(calc(1, 2, 3))    # TypeError: calc() takes 1 positional argument but 3 were given
"""

"""
# 把函数的参数改为可变参数
# 在函数内部，参数numbers接收到的是一个tuple，因此，函数代码完全不变。
# 但是，调用该函数时，可以传入任意个参数，包括0个参数
"""
def calc2(*numbers):
    sum = 0
    for n in numbers:
        sum = sum + n * n
    return sum
print(calc2(1, 2))  # 5
print(calc2())  # 0

# 如果已经有一个list或者tuple，要调用一个可变参数怎么办? 可以用以下方法但比较笨拙
nums = [1, 2, 3]
print(calc2(nums[0], nums[1], nums[2]))  # 14
# 优化办法：Python允许你在list或tuple前面加一个*号，把list或tuple的元素变成可变参数传进去
print(calc2(*nums))  # 14
# *nums表示把nums这个list的所有元素作为可变参数传进去。这种写法相当有用，而且很常见

"""
关键字参数：“含参数名的参数”
    语法：**变量名
    1 可变参数允许你传入0个或任意个参数，这些可变参数在函数调用时自动组装为一个tuple。
    2 关键字参数允许你传入0个或任意个含参数名的参数，这些关键字参数在函数内部自动组装为一个dict。
"""

"""
 函数person除了必选参数name和age外，还接受关键字参数kw。在调用该函数时，可以只传入必选参数
"""


def person(name, age, **kw):
    print('name:', name, 'age:', age, 'other:', kw)


person('Michael', 30)  # name: Michael age: 30 other: {}

# 也可以传入任意个数的关键字参数
person('Bob', 35, city='Beijing')   # name: Bob age: 35 other: {'city': 'Beijing'}

person('Adam', 45, gender='M', job='Engineer')  # name: Adam age: 45 other: {'gender': 'M', 'job': 'Engineer'}

# 和可变参数类似，也可以先组装出一个dict，然后，把该dict转换为关键字参数传进去：
extra = {'city': 'Beijing', 'job': 'Engineer'}
# name: Jack age: 24 other: {'city': 'Beijing', 'job': 'Engineer'}
person('Jack', 24, city=extra['city'], job=extra['job'])

# 上面复杂的调用可以用简化的写法 name: Jack age: 24 other: {'city': 'Beijing', 'job': 'Engineer'}
person('Jack', 24, **extra)

"""
**extra表示把extra这个dict的所有key-value用关键字参数传入到函数的**kw参数，kw将获得一个dict，
注意kw获得的dict是extra的一份拷贝，对kw的改动不会影响到函数外的extra
"""

"""
命名关键字参数:限制关键字参数的名字，“含参数名的参数”
    语法：*,命名关键字参数1，命名关键字参数2

关键字参数的缺点：
      对于关键字参数，函数的调用者可以传入任意不受限制的关键字参数。但是至于到底传入了哪些，就需要在函数内部通过kw检查。
"""

# 仍以person()函数为例，我们希望检查是否有city和job参数
def person(name, age, **kw):
    if 'city' in kw:
        # 有city参数
        pass
    if 'job' in kw:
        # 有job参数
        pass
    print('name:', name, 'age:', age, 'other:', kw)
# name: Jack age: 24 other: {'city': 'Beijing', 'addr': 'Chaoyang', 'zipcode': 123456}
person('Jack', 24, city='Beijing', addr='Chaoyang', zipcode=123456)

"""
# 如果要限制关键字参数的名字，就可以用命名关键字参数，例如，只接收city和job作为关键字参数
# 和关键字参数**kw不同，命名关键字参数需要一个特殊分隔符*，*后面的参数被视为命名关键字参数
"""
def person2(name, age, *, city, job):
    print(name, age, city, job)

person2('Jack', 24, city='Beijing', job='Engineer') # Jack 24 Beijing Engineer

# 如果函数定义中已经有了一个可变参数，后面跟着的命名关键字参数就不再需要一个特殊分隔符*了
def person3(name, age, *args, city, job):
    print(name, age, args, city, job)

"""
# 命名关键字参数必须传入参数名，这和位置参数不同。如果没有传入参数名，调用将报错
                 IndentationError: unindent does not match any outer indentation level
#  
比如：person3('Jack', 24, 'Beijing', 'Engineer')
# 由于调用时缺少参数名city和job，Python解释器把这4个参数均视为位置参数，但person()函数仅接受2个位置参数
# 命名关键字参数可以有缺省值，从而简化调用
"""


def person4(name, age, *, city='Beijing', job):
    print(name, age, city, job)


person('Jack', 24, job='Engineer')  # name: Jack age: 24 other: {'job': 'Engineer'}


#  使用命名关键字参数时，要特别注意，如果没有可变参数，就必须加一个*作为特殊分隔符。
#  如果缺少*，Python解释器将无法识别位置参数和命名关键字参数
def person5(name, age, city, job):
    # 缺少 *，city和job被视为位置参数
    pass


"""
参数组合
在Python中定义函数，可以用必选参数、默认参数、可变参数、关键字参数和命名关键字参数，这5种参数都可以组合使用。
但是请注意，参数定义的顺序必须是：
            (必选参数-->默认参数-->可变参数-->命名关键字参数--->关键字参数)
"""


def f1(a, b, c=0, *args, **kw):
    print('a =', a, 'b =', b, 'c =', c, 'args =', args, 'kw =', kw)


def f2(a, b, c=0, *, d, **kw):
    print('a =', a, 'b =', b, 'c =', c, 'd =', d, 'kw =', kw)


# 函数调用的时候，Python解释器自动按照参数位置和参数名把对应的参数传进去。
f1(1, 2)  # a = 1 b = 2 c = 0 args = () kw = {}
f1(1, 2, c=3)  # a = 1 b = 2 c = 3 args = () kw = {}
f1(1, 2, 3, 'a', 'b')   # a = 1 b = 2 c = 3 args = ('a', 'b') kw = {}
f1(1, 2, 3, 'a', 'b', x=99)  # a = 1 b = 2 c = 3 args = ('a', 'b') kw = {'x': 99}
f2(1, 2, d=99, ext=None)    # a = 1 b = 2 c = 0 d = 99 kw = {'ext': None}


# 最神奇的是通过一个tuple和dict，你也可以调用上述函数
args = (1, 2, 3, 4)
kw = {'d': 99, 'x': '#'}
f1(*args, **kw)  # a = 1 b = 2 c = 3 args = (4,) kw = {'d': 99, 'x': '#'}
args = (1, 2, 3)
kw = {'d': 88, 'x': '#'}
f2(*args, **kw) # a = 1 b = 2 c = 3 d = 88 kw = {'x': '#'}

# 虽然可以组合多达5种参数，但不要同时使用太多的组合，否则函数接口的可理解性很差。

