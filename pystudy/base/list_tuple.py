

"""
list:
1 列表Python内置的一种数据类型
2 list是一种有序的集合，可以随时添加和删除其中的元素
3 list元素是可变的
4 list里面的元素的数据类型也可以不同
"""

# 列出班里所有同学的名字
classmates = ['Michael', 'Bob', 'Tracy']
print(classmates)
classmates.sort()   # 排序 ['Bob', 'Michael', 'Tracy']
print(classmates)

# len()函数可以获得list元素的个数
print(len(classmates))  # 3

# 用索引来访问list中每一个位置的元素，记得索引是从0开始的,不能越界访问
print(classmates[0])  # Michael

# 如果要取最后一个元素，除了计算索引位置外，还可以用-1做索引，直接获取最后一个元素,以此类推，可以获取倒数第2个、倒数第3个
print(classmates[-1])  # Tracy
print(classmates[-2])  # Bob
print(classmates[-3])  # Michael
# 倒数第4个就越界了。
# print(classmates[-4])

# list是一个可变的有序表，所以，可以往list中追加元素到末尾：
classmates.append('Adam')
print(classmates)   # ['Michael', 'Bob', 'Tracy', 'Adam']

# 也可以把元素插入到指定的位置，比如索引号为1的位置：
classmates.insert(1, 'Jack')
print(classmates)   # ['Michael', 'Jack', 'Bob', 'Tracy', 'Adam']

# 要删除list末尾的元素，用pop()方法
classmates.pop()
print(classmates)  # ['Michael', 'Jack', 'Bob', 'Tracy']


# 要删除指定位置的元素，用pop(i)方法，其中i是索引位置
classmates.pop(1)
print(classmates)   # ['Michael', 'Bob', 'Tracy']

# 要把某个元素替换成别的元素，可以直接赋值给对应的索引位置
classmates[1] = 'Sarah'
print(classmates)   # ['Michael', 'Sarah', 'Tracy']

# list里面的元素的数据类型也可以不同
L = ['Apple', 123, True]
# list元素也可以是另一个list
s = ['python', 'java', ['asp', 'php'], 'scheme']
print(len(s))  # 4

# 要注意s只有4个元素，其中s[2]又是一个list，如果拆开写就更容易理解了
p = ['asp', 'php']
s = ['python', 'java', p, 'scheme']
print(s)    # ['python', 'java', ['asp', 'php'], 'scheme']

# 要拿到'php'可以写p[1]或者s[2][1]，因此s可以看成是一个二维数组
print(p[1])
print(s[2][1])

# 如果一个list中一个元素也没有，就是一个空的list，它的长度为0
L = []
print(len(L))   # 0


"""
tuple:
1 tuple和list非常类似，但是tuple一旦初始化就不能修改
2 tuple是另一种有序列表，叫元组
3 因为tuple不可变，所以代码更安全。如果可能，能用tuple代替list就尽量用tuple
"""

classmates = ('Michael', 'Bob', 'Tracy')
"""
这行代码的解释：
1 classmates这个tuple不能变了，它也没有append()，insert()这样的方法。
2 其他获取元素的方法和list是一样的，你可以正常地使用classmates[0]，classmates[-1]，但不能赋值成另外的元素
"""
print(classmates[0])  # Michael

# tuple的陷阱：当你定义一个tuple时，在定义的时候，tuple的元素就必须被确定下来
t = (1, 2)
print(t)

# 如果要定义一个空的tuple，可以写成()
t = ()
print(t)

"""
1个元素的tuple:
错误定义方式: t = (1)
    这是因为括号()既可以表示tuple，又可以表示数学公式中的小括号，这就产生了歧义，
因此，Python规定，这种情况下，按小括号进行计算，计算结果自然是1
所以:
    只有1个元素的tuple定义时必须加一个逗号,，来消除歧义
    Python在显示只有1个元素的tuple时，也会加一个逗号,，以免你误解成数学计算意义上的括号
"""
t = (1,)
print(t)    # (1,)


"""
“可变的”tuple:
    tuple的元素确实变了，但其实变的不是tuple的元素，而是list的元素。
tuple一开始指向的list并没有改成别的list，所以，tuple所谓的“不变”是说，tuple的每个元素，指向永远不变。
即指向'a'，就不能改成指向'b'，指向一个list，就不能改成指向其他对象，但指向的这个list本身是可变的！
如下例子：
"""
t = ('a', 'b', ['A', 'B'])
t[2][0] = 'X'
t[2][1] = 'Y'
print(t)    # ('a', 'b', ['X', 'Y'])



