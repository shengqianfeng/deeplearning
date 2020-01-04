"""
如果错误没有被捕获，它就会一直往上抛，最后被Python解释器捕获，打印一个错误信息，然后程序退出。来看看err.py
"""
# def foo(s):
#     return 10 / int(s)
#
# def bar(s):
#     return foo(s) * 2
#
# def main():
#     bar('0')

# main()


"""
记录错误：Python内置的logging模块可以非常容易地记录错误信息
"""
# import logging
#
# def foo(s):
#     return 10 / int(s)
#
# def bar(s):
#     return foo(s) * 2
#
# def main():
#     try:
#         bar('0')
#     except Exception as e:
#         logging.exception("----", e)
#
# main()
# print('END')
# 通过配置，logging还可以把错误记录到日志文件里，方便事后排查



