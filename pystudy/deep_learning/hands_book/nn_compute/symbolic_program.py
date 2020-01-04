"""
符号式编程
    符号式编程将计算过程抽象为一张计算图(符号图)来描述整个计算过程
    编译易优化、可移植在非python环境下运行，TensorFlow使用了符号式编程
命令式编程：
    容易调试、编写直观

选择：
    大部分深度学习框架在命令式编程和符号式编程之间二选一

"""
def add_str():
    return '''
def add(a, b):
    return a + b
'''

def fancy_func_str():
    return '''
def fancy_func(a, b, c, d):
    e = add(a, b)
    f = add(c, d)
    g = add(e, f)
    return g
'''

def evoke_str():
    return add_str() + fancy_func_str() + '''
print(fancy_func(1, 2, 3, 4))
'''

print("----------------")
prog = evoke_str()
"""
完整打印：
def add(a, b):
    return a + b
def fancy_func(a, b, c, d):
    e = add(a, b)
    f = add(c, d)
    g = add(e, f)
    return
print(fancy_func(1, 2, 3, 4))
"""
print(prog)
print("---------------------")
y = compile(prog,'','exec')  # 通过compile函数编译完整的计算流程并运行
exec(y) # 10