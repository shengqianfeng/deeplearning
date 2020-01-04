
"""
高级语言通常都内置了一套try...except...finally...的错误处理机制，Python也不例外
"""

try:
    print('try...')
    r = 10 / 0
    print('result:', r)
except ZeroDivisionError as e:
    print('except:', e)
finally:
    print('finally...')
print('END')


"""
可以有多个except来捕获不同类型的错误
"""
try:
    print('try...')
    r = 10 / int('a')
    print('result:', r)
except ValueError as e:
    print('ValueError:', e)  # 将被打印 int()函数可能会抛出ValueError，所以我们用一个except捕获ValueError，用另一个except捕获ZeroDivisionError
except ZeroDivisionError as e:
    print('ZeroDivisionError:', e)
finally:
    print('finally...')
print('END')


"""
如果没有错误发生，可以在except语句块后面加一个else，当没有错误发生时，会自动执行else语句
"""
try:
    print('try...')
    r = 10 / int('2')
    print('result:', r)
except ValueError as e:
    print('ValueError:', e)
except ZeroDivisionError as e:
    print('ZeroDivisionError:', e)
else:
    print('no error!')  # 将被打印
finally:
    print('finally...')
print('END')


"""
Python的错误其实也是class，所有的错误类型都继承自BaseException，
所以在使用except时需要注意的是，它不但捕获该类型的错误，还把其子类也“一网打尽”
"""
# try:
#     foo()
# except ValueError as e:
#     print('ValueError')
# except UnicodeError as e:
#     print('UnicodeError')

# 第二个except永远也捕获不到UnicodeError，因为UnicodeError是ValueError的子类，如果有，也被第一个except给捕获了

"""
使用try...except捕获错误还有一个巨大的好处，就是可以跨越多层调用，
比如函数main()调用foo()，foo()调用bar()，结果bar()出错了，这时，只要main()捕获到了，就可以处理
"""

def foo(s):
    return 10 / int(s)

def bar(s):
    return foo(s) * 2

def main():
    try:
        bar('0')
    except Exception as e:
        print('Error:', e)  # Error: division by zero
    finally:
        print('finally...')


main()



