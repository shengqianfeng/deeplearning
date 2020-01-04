

class Student(object):

    def __init__(self, name, score):
        self.__name__ = name    # 定义为public
        self.__score = score    # 定义为privatee

    def print_score(self):
        print('%s: %s' % (self.__name, self.__score))

    def get_name(self):
        return self.__name__

    def get_score(self):
        return self.__score


s = Student("JeffSheng", "99")
print("Student type is:", type(Student))     # Student type is: <class 'type'>
print(type(s))  # <class '__main__.Student'>
score = s.get_score()
print("type(score) is :", type(score))  # type(score) is : <class 'str'>

print(s.get_score())         # 99
print(s._Student__score)     # 99
# print(s.__score)     # 报错：AttributeError: 'Student' object has no attribute '__score'

print(s.__name__)   # JeffSheng
print(s.get_name())   # JeffSheng


class Animal(object):
    def run(self):
        print('Animal is running...')


class Dog(Animal):
    def run(self):
        print('Dog is running...')

    def eat(self):
        print('Eating meat...')


class Cat(Animal):
    def run(self):
        print('Cat is running...')

class Husky(Dog):
    def run(self):
        print('Husky is running...')


dog = Dog()
dog.run()   # Dog is running...
cat = Cat()
cat.run()   # Cat is running...
husky = Husky()
husky.run() # Husky is running...

# a = list() # a是list类型
# b = Animal() # b是Animal类型
# c = Dog() # c是Dog类型
# # 判断一个变量是否是某个类型可以用isinstance()判断
# print(isinstance(a, list))  #True
# print(isinstance(b, Animal))    #True
# print(isinstance(c, Dog))   #True

# print(isinstance(c, Animal))   #c既是Dog也是Animal类型 True
"""
所以，在继承关系中，如果一个实例的数据类型是某个子类，那它的数据类型也可以被看做是父类,但是反过来不行。
"""



def run_twice(animal):
    animal.run()
    animal.run()


run_twice(Animal()) # 打印两次Animal is running...

run_twice(Dog())    # 打印两次Dog is running...

run_twice(Cat())  # 打印两次Cat is running

class Tortoise(Animal):
    def run(self):
        print('Tortoise is running slowly...')


run_twice(Tortoise())  # 打印两次Tortoise is running slowly...

"""
静态语言 vs 动态语言
对于静态语言（例如Java）来说，如果需要传入Animal类型，则传入的对象必须是Animal类型或者它的子类，否则，将无法调用run()方法
对于Python这样的动态语言来说，则不一定需要传入Animal类型。我们只需要保证传入的对象有一个run()方法就可以了
"""
class Timer(object):
    def run(self):
        print('Start...')


run_twice(Timer())  # 打印两次Start...



# 判断对象类型，使用type()函数
print(type(123))    # <class 'int'>
print( type('str')) #<class 'str'>
print(type(None))   # <class 'NoneType'>

# 如果一个变量指向函数或者类，也可以用type()判断
print( type(abs))   # <class 'builtin_function_or_method'>
print(type(dog))    # <class '__main__.Dog'>
import numpy as np
print( type(123)==type(456))    # True
print( type(123)==int)  # True
print( type('abc')==type('123'))    # True
print(type('abc')==str)     # True
print(type('abc')==type(123))   # False


print("----------arange type--------------")
a = np.arange(-1, 1, 0.5)
print(a)    # [-1.  -0.5  0.   0.5]
print(type(a))  # <class 'numpy.ndarray'>
print("----------array type--------------")
a = [-1,-0.5,0,0.5] # [-1, -0.5, 0, 0.5, 1]
print(a)
print(type(a))  # <class 'list'>


"""
如果要判断一个对象是否是函数怎么办？可以使用types模块中定义的常量
"""
import types


def fn():
    pass


print(type(fn) == types.FunctionType)   # True
print(type(abs)==types.BuiltinFunctionType)    # True
print(type(lambda x: x)==types.LambdaType)  # True
print(type((x for x in range(10))) == types.GeneratorType)  # True

a = Animal()
d = Dog()
h = Husky()
# isinstance()就可以告诉我们，一个对象是否是某种类型
print(isinstance(h, Husky)) # True
print("h is dog?", isinstance(h, Dog))  # h is dog? True

# isinstance()判断的是一个对象是否是该类型本身，或者位于该类型的父继承链上
print(isinstance(h, Animal))    # True
print(isinstance(d, Dog) and isinstance(d, Animal))  # True
print(isinstance(d, Husky))  # False d不是Husky类型

# 能用type()判断的基本类型也可以用isinstance()判断
print(isinstance('a', str))  # True
print(isinstance(123, int))     # True
print(isinstance(b'a', bytes))  # True
# 并且还可以判断一个变量是否是某些类型中的一种
print(isinstance([1, 2, 3], (list, tuple))) # True
print(isinstance((1, 2, 3), (list, tuple))) # True



"""
dir函数
    如果要获得一个对象的所有属性和方法，可以使用dir()函数，它返回一个包含字符串的list，比如，获得一个str对象的所有属性和方法
"""
print(dir('ABC'))

# 类似__xxx__的属性和方法在Python中都是有特殊用途的，比如__len__方法返回长度。
# 在Python中，如果你调用len()函数试图获取一个对象的长度，实际上，在len()函数内部，它自动去调用该对象的__len__()方法，
# 所以，下面的代码是等价的
print(len('ABC'))   # 3
print('ABC'.__len__())  # 3

print('ABC'.lower())    # abc


"""
配合getattr()、setattr()以及hasattr()，我们可以直接操作一个对象的状态
"""
class MyObject(object):
    def __init__(self):
        self.x = 9

    def power(self):
        return self.x * self.x


obj = MyObject()
print(hasattr(obj, 'x'))  # 有属性'x'吗？True
print(obj.x)    # 9

print(hasattr(obj, 'y'))  # 有属性'y'吗？)False
setattr(obj, 'y', 19)   # # 设置一个属性'y'
print(hasattr(obj, 'y'))  # 有属性'y'吗？ True
print(getattr(obj, 'y'))  # 获取属性'y' 19

print(obj.y)  # 获取属性'y' 19

# 如果试图获取不存在的属性，会抛出AttributeError的错误
# 可以传入一个default参数，如果属性不存在，就返回默认值
print(getattr(obj, 'z', 404))   # 获取属性'z'，如果不存在，返回默认值404


"""
也可以获得对象的方法
"""
print(hasattr(obj, 'power'))   # 有属性'power'吗？True
# <bound method MyObject.power of <__main__.MyObject object at 0x000001F7291554C8>>
print(getattr(obj, 'power'))  # 获取属性'power'

fn = getattr(obj, 'power')  # 获取属性'power'并赋值到变量fn
print(fn())

"""
由于Python是动态语言，根据类创建的实例可以任意绑定属性:给实例绑定属性的方法是通过实例变量，或者通过self变量
"""
class Student(object):
    def __init__(self, name):
        self.name = name


s = Student('Bob')
s.score = 90

"""
如果Student类本身需要绑定一个属性呢？
    1 可以直接在class中定义属性，这种属性是类属性，归Student类所有
    2 当我们定义了一个类属性后，这个属性虽然归类所有，但类的所有实例都可以访问到
"""


class Student(object):
    name = 'Student'

s = Student()  # 创建实例s
print(s.name)   # 打印name属性，因为实例并没有name属性，所以会继续查找class的name属性
print(Student.name)     # 打印类的name属性
s.name = 'Michael'  # 给实例绑定name属性
print(s.name)  # Michael 由于实例属性优先级比类属性高，因此，它会屏蔽掉类的name属性
print(Student.name)  # Student 但是类属性并未消失，用Student.name仍然可以访问

del s.name  # 如果删除实例的name属性
print(s.name)  # Student 再次调用s.name，由于实例的name属性没有找到，类的name属性就显示出来了

class Teacher(object):
    def __init__(self, name):
        self.name = name
        self.__other_name = name
        self.__book_name__ = name

    def jiangke(self):
        print("我正在讲课.....")












