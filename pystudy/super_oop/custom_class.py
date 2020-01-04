"""
Python的class中还有许多这样有特殊用途的函数，可以帮助我们定制类
"""

# __str__
class Student(object):
    def __init__(self, name):
        self.name = name


print(Student('Michael'))
# 以下打印不好看，怎么才能打印得好看呢？只需要定义好__str__()方法，返回一个好看的字符串就可以了
#  <__main__.Student object at 0x00000163B86F6108>

class Student(object):
    def __init__(self, name):
        self.name = name

    def __str__(self):
        return 'Student object (name: %s)' % self.name


print(Student('Michael'))   # Student object (name: Michael)

"""
直接敲变量不用print，打印出来的实例还是不好看,直接显示变量调用的不是__str__()，而是__repr__()
__str__()返回用户看到的字符串，而__repr__()返回程序开发者看到的字符串，也就是说，__repr__()是为调试服务的
解决办法是再定义一个__repr__()。但是通常__str__()和__repr__()代码都是一样的，所以，有个偷懒的写法
"""
class Student(object):
    def __init__(self, name):
        self.name = name
    def __str__(self):
        return 'Student object (name=%s)' % self.name
    __repr__ = __str__


"""
 __iter__:
    如果一个类想被用于for ... in循环，类似list或tuple那样，就必须实现一个__iter__()方法，该方法返回一个迭代对象，
 然后，Python的for循环就会不断调用该迭代对象的__next__()方法拿到循环的下一个值，直到遇到StopIteration错误时退出循环
"""
# 们以斐波那契数列为例，写一个Fib类，可以作用于for循环
class Fib(object):
    def __init__(self):
        self.a, self.b = 0, 1  # 初始化两个计数器a，b

    def __iter__(self):
        return self  # 实例本身就是迭代对象，故返回自己

    def __next__(self):
        self.a, self.b = self.b, self.a + self.b  # 计算下一个值
        if self.a > 100000:     # 退出循环的条件
            raise StopIteration()
        return self.a   # 返回下一个值


# 试试把Fib实例作用于for循环
for n in Fib():
    print(n)




"""
__getitem__
    Fib实例虽然能作用于for循环，看起来和list有点像，但是，把它当成list来使用还是不行，比如，取第5个元素
    要表现得像list那样按照下标取出元素，需要实现__getitem__()方法
"""
class Fib(object):
    def __getitem__(self, n):
        a, b = 1, 1
        for x in range(n):
            a, b = b, a + b
        return a


# 现在，就可以按下标访问数列的任意一项了
f = Fib()
print(f[10])    # 89


"""
但是list有个神奇的切片方法
对于Fib却报错。原因是__getitem__()传入的参数可能是一个int，也可能是一个切片对象slice，所以要做判断
"""
class Fib(object):
    def __getitem__(self, n):
        if isinstance(n, int):  # n是索引
            a, b = 1, 1
            for x in range(n):
                a, b = b, a + b
            return a
        if isinstance(n, slice):    # n是切片
            start = n.start
            stop = n.stop
            if start is None:
                start = 0
            a, b = 1, 1
            L = []
            for x in range(stop):
                if x >= start:
                    L.append(a)
                a, b = b, a + b
            return L


f = Fib()
print(f[0:5])   # [1, 1, 2, 3, 5]



"""
__getattr__
    正常情况下，当我们调用类的方法或属性时，如果不存在，就会报错。比如定义Student
"""
class Student(object):
    def __init__(self):
        self.name = 'Michael'
# 调用name属性，没问题，但是，调用不存在的score属性，就有问题了
s = Student()
print(s.name)
# print(s.score)      # AttributeError: 'Student' object has no attribute 'score'

"""
要避免这个错误，除了可以加上一个score属性外，Python还有另一个机制，那就是写一个__getattr__()方法，动态返回一个属性
当调用不存在的属性时，比如score，Python解释器会试图调用__getattr__(self, 'score')来尝试获得属性，这样，我们就有机会返回score的值
"""
class Student(object):
    def __init__(self):
        self.name = 'Michael'
    def __getattr__(self, attr):
        if attr=='score':
            return 99


s = Student()
print(s.name)   # Michael
print(s.score)  # 99


# 返回函数也是完全可以的 只是调用方式要变为 s.age()
class Student(object):
    def __getattr__(self, attr):
        if attr == 'age':
            return lambda: 25
s = Student()
print(s.age())  # 25

"""
注意到任意调用如s.abc都会返回None，这是因为我们定义的__getattr__默认返回就是None。要让class只响应特定的几个属性，我们就要按照约定，抛出AttributeError的错误
"""
class Student(object):
    def __getattr__(self, attr):
        if attr == 'age':
            return lambda: 25
        raise AttributeError('\'Student\' object has no attribute \'%s\'' % attr)

s = Student()
# print(s.age1())  # AttributeError: 'Student' object has no attribute 'age1'


class Chain(object):

    def __init__(self, path=''):
        self._path = path

    def __getattr__(self, path):
        return Chain('%s/%s' % (self._path, path))

    def __str__(self):
        return self._path

    __repr__ = __str__


# /status/user/timeline/list
print(Chain().status.user.timeline.list)


"""
__call__
    一个对象实例可以有自己的属性和方法，当我们调用实例方法时，我们用instance.method()来调用。能不能直接在实例本身上调用呢？在Python中，答案是肯定的
"""
# 任何类，只需要定义一个__call__()方法，就可以直接对实例进行调用
class Student(object):
    def __init__(self, name):
        self.name = name

    def __call__(self):
        print('My name is %s.' % self.name)


s = Student('Michael')
s()  # self参数不要传入 My name is Michael.

"""
__call__()还可以定义参数。对实例进行直接调用就好比对一个函数进行调用一样，所以你完全可以把对象看成函数，把函数看成对象，因为这两者之间本来就没啥根本的区别
如果你把对象看成函数，那么函数本身其实也可以在运行期动态创建出来，因为类的实例都是运行期创建出来的，这么一来，我们就模糊了对象和函数的界限。
那么，怎么判断一个变量是对象还是函数呢？
    其实，更多的时候，我们需要判断一个对象是否能被调用，能被调用的对象就是一个Callable对象，比如函数和我们上面定义的带有__call__()的类实例
"""
print(callable(Student('Michael')))  # True
print(callable(max))    # True
print(callable([1, 2, 3]))  # False
print(callable(None))   # False
print(callable('str'))  # False

# 通过callable()函数，我们就可以判断一个对象是否是“可调用”对象。





