age = 20
if age >= 18:
    print('your age is', age)
    print('adult')


age = 3
if age >= 18:
    print('your age is', age)
    print('adult')
else:
    print('your age is', age)
    print('teenager')


age = 3
if age >= 18:
    print('adult')
elif age >= 6:
    print('teenager')
else:
    print('kid')

# elif是else if的缩写，完全可以有多个elif
age = 20
if age >= 6:
    print('teenager')
elif age >= 18:
    print('adult')
else:
    print('kid')

# if判断条件还可以简写
# 只要x是非零数值、非空字符串、非空list等，就判断为True，否则为False。
# if x:
#     print('True')

"""
input()返回的数据类型是str，str不能直接和整数比较，必须先把str转换成整数

"""
birth = input('birth: ')
# 如果输入abc就会报错：ValueError: invalid literal for int() with base 10: 'abc'
birth = int(birth)
if birth < 2000:
    print('00前')
else:
    print('00后')

