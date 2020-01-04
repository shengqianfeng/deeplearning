"""
Py操作文件和目录
其实操作系统提供的命令只是简单地调用了操作系统提供的接口函数，Python内置的os模块也可以直接调用操作系统提供的接口函数
"""
import os

print(os.name)  # 操作系统类型 nt  如果是posix，说明系统是Linux、Unix或Mac OS X，如果是nt，就是Windows系统
# print(os.uname()) # windows不提供


"""
查看环境变量
在操作系统中定义的环境变量，全部保存在os.environ这个变量中，可以直接查看
"""
print(os.environ)
# 要获取某个环境变量的值，可以调用os.environ.get('key')
print(os.environ.get('PATH'))



"""
操作文件和目录
操作文件和目录的函数一部分放在os模块中，一部分放在os.path模块中，这一点要注意一下。查看、创建和删除目录可以这么调用
"""
# 把两个路径合成一个时，不要直接拼字符串，而要通过os.path.join()函数，这样可以正确处理不同操作系统的路径分隔符。
print(os.path.abspath('.'))  # 查看当前目录的绝对路径

# 在某个目录下创建一个新目录，首先把新目录的完整路径表示出来:
print(os.path.join('D://', 'testdir'))
if os.path.exists('D://testdir')==None:
    print(os.mkdir('D://testdir'))  # 创建目录

if os.path.exists('D://testdir'):
    print(os.rmdir('D://testdir'))  # 删除目录

print([x for x in os.listdir('..') if os.path.isdir(x)])
# 要列出所有的.py文件
print([x for x in os.listdir('.') if os.path.isfile(x) and os.path.splitext(x)[1]=='.py'])

# listdir()函数不可对目录的子目录进行扫描
print(os.listdir('..'))

"""
# 很多时候我们需要将某个文件夹下的所有文件都要找出来，那么此时我们就需要os.walk()函数。
# os.walk()方法用于通过在目录了数中向下或向上游走，输出目录的文件名，该方法回访回一个生成器
语法：os.walk(top[, topdown=True[, onerror=None[, followlinks=False]]])

    top 	-- 是你所要遍历的目录的地址.
    topdown -- 可选,为 True,则优先遍历top目录,
                 否则False，优先遍历 top 的子目录(默认为开启)。
    onerror -- 可选，需要一个 callable 回调对象,当 walk 需要异常时，会调用。
    followlinks -- 可选,如果为 True，则会遍历目录下的快捷方式，默认开启 
    return None 返回值
         该函数没有返回值会使用yield关键字抛出一个存放当前该层目录(root,dirs,files)的三元组，最终将所有目录层的的结果变为一个生成器
               root  所指的是当前正在遍历的这个文件夹的本身的地址
               dirs  是一个 list ，内容是该文件夹中所有的目录的名字(不包括子目录)
               files 同样是 list , 内容是该文件夹中所有的文件(不包括子目录)	
"""
print(list(os.walk('../grpc',topdown = True))) #优先遍历../grpc目录
"""
这个列表（因为我们将生成器强制转化为列表）中的每一个元素都是一个元组，每一个元组都是都是一层目录，
元组中的元素又代表了:
    1 当前目录层的路径(根据赋给形参top的值来决定是绝对路径还是相对路径)
    2 当前目录下有的文件目录
    3 和当前目录下的文件
"""
# 当我们对生成器使用for循环时，会将每层一目录信息返回
for info in os.walk('../grpc', topdown=True):
    print(info)
    print('****************************')



