import atexit


"""

使用atexit模块在程序退出时执行一些清理工作
"""

def atexitFunc_1():
    print('I am atexitFunc_1')


def atexitFunc_2(name, age):
    print('I am atexitFunc_2, %s is %d years old' % (name, age))


print('I am the first output')

atexit.register(atexitFunc_1)
atexit.register(atexitFunc_2, 'Katherine', 20)


# 也可以使用装饰器语法
@atexit.register
def atexitFunc_3():
    print('I am atexitFunc_3')
