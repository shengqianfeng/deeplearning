"""
调试：
1 打印print()
2 assert断言
    启动Python解释器时可以用-O参数来关闭assert,关闭后，你可以把所有的assert语句当成pass来看
3 logging
    和assert比，logging不会抛出错误，而且可以输出到文件
    logging.info()就可以输出一段文本
    这就是logging的好处，它允许你指定记录信息的级别，有debug，info，warning，error等几个级别，当我们指定level=INFO时，logging.debug就不起作用了。同理，指定level=WARNING后，debug和info就不起作用了。
    logging的另一个好处是通过简单的配置，一条语句可以同时输出到不同的地方，比如console和文件

4 pdb
    第4种方式是启动Python的调试器pdb，让程序以单步方式运行，可以随时查看运行状态
"""

# assert的意思是，表达式n != 0应该是True，否则，根据程序运行的逻辑，后面的代码肯定会出错
# 如果断言失败，assert语句本身就会抛出AssertionError
# def foo(s):
#     n = int(s)
#     assert n != 0, 'n is zero!'     # AssertionError: n is zero!
#     return 10 / n
#
# def main():
#     foo('0')
#
# # main()


# import logging
# logging.basicConfig(level=logging.INFO)
#
# s = '0'
# n = int(s)
# logging.info('n = %d' % n)  # INFO:root:n = 0
# print(10 / n)


# s = '0'
# n = int(s)
# print(10 / n)



import pdb

s = '0'
n = int(s)
pdb.set_trace()  # 运行到这里会自动暂停
print(10 / n)



