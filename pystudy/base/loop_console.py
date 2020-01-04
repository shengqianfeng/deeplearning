
"""
循环控制

"""
print(1 + 2 + 3)


# Python的循环有两种，一种是for...in循环，依次把list或tuple中的每个元素迭代出来
# 遍历list
names = ['Michael', 'Bob', 'Tracy']
for name in names:
    print(name)


# 累加和
sum = 0
for x in [1, 2, 3, 4, 5, 6, 7, 8, 9, 10]:
    sum = sum + x
print(sum)

# Python提供一个range()函数，可以生成一个整数序列，再通过list()函数可以转换为list
# range(5)生成的序列是从0开始小于5的整数
print(list(range(5)))   # [0, 1, 2, 3, 4]


# Python提供的第二种循环是while循环，只要条件满足，就不断循环，条件不满足时退出循环
sum = 0
n = 99
while n > 0:
    sum = sum + n
    n = n - 2
print(sum)  # 2500

# 在循环中，break语句可以提前退出循环。
n = 1
while n <= 100:
    if n > 10:  # 当n = 11时，条件满足，执行break语句
        break  # break语句会结束当前循环
    print(n)
    n = n + 1
print('END')


# 循环过程中，也可以通过continue语句，跳过当前的这次循环，直接开始下一次循环
n = 0
while n < 10:
    n = n + 1
    if n % 2 == 0:  # 如果n是偶数，执行continue语句
        continue  # continue语句会直接继续下一轮循环，后续的print()语句不会执行
    print(n)

