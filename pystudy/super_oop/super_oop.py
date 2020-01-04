"""
正常情况下，当我们定义了一个class，创建了一个class的实例后，我们可以给该实例绑定任何属性和方法，这就是动态语言的灵活性。
"""


class Student(object):
    pass


s = Student()
s.name = 'Michael'  # 动态给实例绑定一个属性
print(s.name)   # Michael

# 定义一个函数作为实例方法
def set_age(self, age):
    self.age = age


from types import MethodType

# 给实例绑定一个方法
s.set_age = MethodType(set_age, s)
s.set_age(25)  # 调用实例方法
print(s.age)    # 测试结果 25

"""
但是，给一个实例绑定的方法，对另一个实例是不起作用的
"""
s2 = Student()  # 创建新的实例
# s2.set_age(25)  # 尝试调用方法 AttributeError: 'Student' object has no attribute 'set_age'


# 为了给所有实例都绑定方法，可以给class绑定方法
def set_score(self, score):
    self.score = score


Student.set_score = set_score
"""
给class绑定方法后，所有实例均可调用
"""
s.set_score(100)
print(s.score)  # 100
s2.set_score(99)
print(s2.score) # 99

"""
通常情况下，上面的set_score方法可以直接定义在class中，但动态绑定允许我们在程序运行的过程中动态给class加上功能，这在静态语言中很难实现
"""


"""
Slots的使用
    只允许对Student实例添加name和age属性
"""


# 为了达到限制的目的，Python允许在定义class的时候，定义一个特殊的__slots__变量，来限制该class实例能添加的属性
class Student(object):
    pass  # __slots__ = ('name', 'age')  # 用tuple定义允许绑定的属性名称


s = Student() # 创建新的实例
s.name = 'Michael'  # 绑定属性'name'
s.age = 25  # 绑定属性'age'
# s.score = 99  # 绑定属性'score' 报错：AttributeError: 'Student' object has no attribute 'score'

"""
使用__slots__要注意，__slots__定义的属性仅对当前类实例起作用，对继承的子类是不起作用的
除非在子类中也定义__slots__，这样，子类实例允许定义的属性就是自身的__slots__加上父类的__slots__
"""
class GraduateStudent(Student):
    pass


g = GraduateStudent()
g.score = 9999
print(g.score)  # 9999



"""
在绑定属性时，如果我们直接把属性暴露出去，虽然写起来很简单，但是，没办法检查参数，导致可以把成绩随便改
这显然不合逻辑。为了限制score的范围，可以通过一个set_score()方法来设置成绩，再通过一个get_score()来获取成绩，这样，在set_score()方法里，就可以检查参数
"""
s = Student()
s.score = 9999

class Student(object):

    def get_score(self):
         return self._score

    def set_score(self, value):
        if not isinstance(value, int):
            raise ValueError('score must be an integer!')
        if value < 0 or value > 100:
            raise ValueError('score must between 0 ~ 100!')
        self._score = value


# 现在，对任意的Student实例进行操作，就不能随心所欲地设置score了
s = Student()
s.set_score(60)  # ok!
print(s.get_score())  # 60
# 发出警告
# This inspection warns if a protected member is accessed outside the class,
# a descendant of the class where it's defined or a module.
print(s._score)  # 60
# print(s.set_score(9999))  # 报错！  ValueError: score must between 0 ~ 100!


# Python内置的@property装饰器就是负责把一个方法变成属性调用的
class Student(object):
    # 把一个getter方法变成属性，只需要加上@property就可以了
    @property
    def score(self):
        return self._score

    @score.setter
    def score(self, value):
        if not isinstance(value, int):
            raise ValueError('score must be an integer!')
        if value < 0 or value > 100:
            raise ValueError('score must between 0 ~ 100!')
        self._score = value

"""
此时，@property本身又创建了另一个装饰器@score.setter，负责把一个setter方法变成属性赋值，于是，我们就拥有一个可控的属性操作
"""
s = Student()
s.score = 60  # OK，实际转化为s.set_score(60)
print("----", s.score)  # OK，实际转化为s.get_score()
# ValueError: score must between 0 ~ 100!
# s.score = 9999

# 还可以定义只读属性，只定义getter方法，不定义setter方法就是一个只读属性
class Student(object):

    @property
    def birth(self):
        return self._birth

    @birth.setter
    def birth(self, value):
        self._birth = value

    @property
    def age(self):
        return 2015 - self._birth
# birth是可读写属性，而age就是一个只读属性，因为age可以根据birth和当前时间计算出来
s = Student()
s.birth=1990
print(s.age)    # 25
# s.age=26    # AttributeError: can't set attribute


