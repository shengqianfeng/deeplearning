#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
正是有了 Pandas 工具，Python 做数据挖掘才具有优势。
@File : _pandas.py
@Author : jeffsheng
@Date : 2019/11/6
@Desc : Pandas 可以说是基于 NumPy 构建的含有更高级数据结构和分析能力的工具包
 Series 和 DataFrame 这两个核心数据结构，他们分别代表着一维的序列和二维的表结构
 基于这两种数据结构，Pandas 可以对数据进行导入、清洗、处理、统计和输出。
"""
"""
Series 是个定长的字典序列。
说是定长是因为在存储的时候，相当于两个 ndarray，这也是和字典结构最大的不同。因为在字典的结构里，元素的个数是不固定的

Series 有两个基本属性：index 和 values。
在 Series 结构中，index 默认是 0,1,2,……递增的整数序列，当然我们也可以自己来指定索引，比如 index=[‘a’, ‘b’, ‘c’, ‘d’]。
"""

import pandas as pd
from pandas import Series, DataFrame


x1 = Series([1,2,3,4])
x2 = Series(data=[1,2,3,4], index=['a', 'b', 'c', 'd'])
"""
0    1
1    2
2    3
3    4
"""
print(x1)
"""
a    1
b    2
c    3
d    4
"""
print(x2)

"""
我们也可以采用字典的方式来创建 Series
"""

d = {'a':1, 'b':2, 'c':3, 'd':4}
x3 = Series(d)
"""
a    1
b    2
c    3
d    4
"""
print(x3)



"""
DataFrame 类型数据结构类似数据库表
它包括了行索引和列索引，我们可以将 DataFrame 看成是由相同索引的 Series 组成的字典类型。
"""
data = {'Chinese': [66, 95, 93, 90,80],'English': [65, 85, 92, 88, 90],'Math': [30, 98, 96, 77, 90]}
df1= DataFrame(data)
df2 = DataFrame(data, index=['ZhangFei', 'GuanYu', 'ZhaoYun', 'HuangZhong', 'DianWei'], columns=['English', 'Math', 'Chinese'])
"""
  Chinese  English  Math
0       66       65    30
1       95       85    98
2       93       92    96
3       90       88    77
4       80       90    90
"""
print(df1)
"""
           English  Math  Chinese
ZhangFei         65    30       66
GuanYu           85    98       95
ZhaoYun          92    96       93
HuangZhong       88    77       90
DianWei          90    90       80
"""
print(df2)



"""
数据导入和输出Pandas 允许直接从 xlsx，csv 等文件中导入数据，也可以输出到 xlsx, csv 等文件，非常方便。
"""
# score = DataFrame(pd.read_excel('data.xlsx'))
# score.to_excel('data1.xlsx')
"""
  name  age sex
0   张三   25   男
1   李四   35   女
2   王五   45   男
"""
# print(score)


"""
数据清洗
"""
# print("----删除chinese前....")
"""
            English  Math  Chinese
ZhangFei         65    30       66
GuanYu           85    98       95
ZhaoYun          92    96       93
HuangZhong       88    77       90
DianWei          90    90       80
"""
# print(df2)
# 删除 DataFrame 中的不必要的列或行
# df2 = df2.drop(columns=['Chinese'])
# print("----删除chinese后....")
"""
           English  Math
ZhangFei         65    30
GuanYu           85    98
ZhaoYun          92    96
HuangZhong       88    77
DianWei          90    90
"""
# print(df2)
# df2 = df2.drop(index=['ZhangFei'])
# print("----删除ZhangFei后....")
"""
            English  Math
GuanYu           85    98
ZhaoYun          92    96
HuangZhong       88    77
DianWei          90    90
"""
print(df2)



"""
重命名列名 columns，让列表名更容易识别
如果你想对 DataFrame 中的 columns 进行重命名，
可以直接使用 rename(columns=new_names, inplace=True) 函数，
比如我把列名 Chinese 改成 YuWen，English 改成 YingYu。
"""
# df2.rename(columns={'Chinese': 'YuWen', 'English': 'Yingyu'}, inplace = True)
# print("-------重命名列名--------")
"""
            Yingyu  Math
GuanYu          85    98
ZhaoYun         92    96
HuangZhong      88    77
DianWei         90    90
"""
print(df2)

"""
去重复的值
数据采集可能存在重复的行，这时只要使用 drop_duplicates() 就会自动把重复的行去掉
"""
# df = df.drop_duplicates() #去除重复行


"""
更改数据格式
"""
# 很多时候数据格式不规范，我们可以使用 astype 函数来规范数据格式，比如我们把 Chinese 字段的值改成 str 类型，或者 int64
import numpy as np
df2['Chinese'].astype('str')
df2['Chinese'].astype(np.int64)

"""
数据间的空格
有时候我们先把格式转成了 str 类型，是为了方便对数据进行操作，这时想要删除数据间的空格，我们就可以使用 strip 函数
"""

#删除左右两边空格
# df2['Chinese']=df2['Chinese'].map(str.strip)
#删除左边空格
# df2['Chinese']=df2['Chinese'].map(str.lstrip)
#删除右边空格
# df2['Chinese']=df2['Chinese'].map(str.rstrip)

"""
如果数据里有某个特殊的符号，我们想要删除怎么办？同样可以使用 strip 函数，比如 Chinese 字段里有美元符号，我们想把这个删掉，可以这么写
"""
# df2['Chinese']=df2['Chinese'].str.strip('$')


"""
大小写转换
大小写是个比较常见的操作，比如人名、城市名等的统一都可能用到大小写的转换，在 Python 里直接使用 upper(), lower(), title() 函数
"""

#全部大写
df2.columns = df2.columns.str.upper()
print(df2)
#全部小写
df2.columns = df2.columns.str.lower()
print(df2)
#首字母大写
df2.columns = df2.columns.str.title()
print(df2)


"""
查找空值
数据量大的情况下，有些字段存在空值 NaN 的可能，这时就需要使用 Pandas 中的 isnull 函数进行查找。比如，我们输入一个数据表如下
"""
data = {'Chinese': [66, 95, 93, 90,80],'English': [65, 85, 92, 88, 90],'Math': [30, 98, 96, 77, None]}
df1= DataFrame(data, index=['ZhangFei', 'GuanYu', 'ZhaoYun', 'HuangZhong', 'DianWei'], )
"""
            Chinese  English  Math
ZhangFei         66       65  30.0
GuanYu           95       85  98.0
ZhaoYun          93       92  96.0
HuangZhong       90       88  77.0
DianWei          80       90   NaN
"""
print(df1)
"""
            Chinese  English   Math
ZhangFei      False    False  False
GuanYu        False    False  False
ZhaoYun       False    False  False
HuangZhong    False    False  False
DianWei       False    False   True
"""
print(df1.isnull())
# 如果我想知道哪列存在空值，可以使用 df.isnull().any()
"""
Chinese    False
English    False
Math        True
dtype: bool
"""
print(df1.isnull().any())


"""
使用 apply 函数对数据进行清洗
apply 函数是 Pandas 中自由度非常高的函数，使用频率也非常高。
"""
# 比如我们想对 name 列的数值都进行大写转化可以用：
data = {'hobby': ['eat', 'play', 'look', 'run','walk']}
df1= DataFrame(data, index=['ZhangFei', 'GuanYu', 'ZhaoYun', 'HuangZhong', 'DianWei'], )
df1['hobby'] = df1['hobby'].apply(str.upper)
"""
           hobby
ZhangFei     EAT
GuanYu      PLAY
ZhaoYun     LOOK
HuangZhong   RUN
DianWei     WALK
"""
print(df1)


# 我们也可以定义个函数，在 apply 中进行使用。
# 比如定义 doing_df 函数是将原来的str+ing 进行返回。
# 可以写成

def doing_df(x):
           return x+'ing'
df1[u'hobby'] = df1[u'hobby'].apply(doing_df)
"""
              hobby
ZhangFei     EATing
GuanYu      PLAYing
ZhaoYun     LOOKing
HuangZhong   RUNing
DianWei     WALKing
"""
print(df1)

"""
我们也可以定义更复杂的函数，比如对于 DataFrame，我们新增两列，
其中’new1’列是“语文”和“英语”成绩之和的 m 倍，'new2’列是“语文”和“英语”成绩之和的 n 倍
"""

def plus(df,n,m):
    df['new1'] = df[u'hobby']+' is doing '
    df['new2'] = df[u'hobby']+'  has get it'
    return df
"""
其中 axis=1 代表按照行为轴进行操作，axis=0 代表按照列为轴进行操作，
args 是传递的两个参数，即 n=2, m=3，在 plus 函数中使用到了 n 和 m，从而生成新的 df
"""
df1 = df1.apply(plus, axis=1, args=(2,3,))
print(df1)


"""
数据统计
在数据清洗后，我们就要对数据进行统计了。
Pandas 和 NumPy 一样，都有常用的统计函数，如果遇到空值 NaN，会自动排除。
"""
# 统计0,1,2,3,4
df1 = DataFrame({'name':['ZhangFei', 'GuanYu', 'a', 'b', 'c'], 'data':range(5)})
# 一次性输出多个统计指标
print (df1.describe())


"""
数据表合并
有时候我们需要将多个渠道源的多个数据表进行合并，
一个 DataFrame 相当于一个数据库的数据表，那么多个 DataFrame 数据表的合并就相当于多个数据库的表合并
"""

df1 = DataFrame({'name':['ZhangFei', 'GuanYu', 'a', 'b', 'c'], 'data1':range(5)})
df2 = DataFrame({'name':['ZhangFei', 'GuanYu', 'A', 'B', 'C'], 'data2':range(5)})
"""
       name  data1
0  ZhangFei      0
1    GuanYu      1
2         a      2
3         b      3
4         c      4
"""
print(df1)

# 两个 DataFrame 数据表的合并使用的是 merge() 函数，有下面 5 种形式
# 1. 基于指定列进行连接
df3 = pd.merge(df1, df2, on='name')
"""
       name  data1  data2
0  ZhangFei      0      0
1    GuanYu      1      1
"""
print(df3)

# 2. inner 内连接
# inner 内链接是 merge 合并的默认情况，inner 内连接其实也就是键的交集，在这里 df1, df2 相同的键是 name，所以是基于 name 字段做的连接：
df3 = pd.merge(df1, df2, how='inner')
"""
       name  data1  data2
0  ZhangFei      0      0
1    GuanYu      1      1
"""
print(df3)

# 3. left 左连接
# 左连接是以第一个 DataFrame 为主进行的连接，第二个 DataFrame 作为补充
df3 = pd.merge(df1, df2, how='left')
"""
       name  data1  data2
0  ZhangFei      0    0.0
1    GuanYu      1    1.0
2         a      2    NaN
3         b      3    NaN
4         c      4    NaN
"""
print(df3)

# 4. right 右连接
# 右连接是以第二个 DataFrame 为主进行的连接，第一个 DataFrame 作为补充。
df3 = pd.merge(df1, df2, how='right')
"""
       name  data1  data2
0  ZhangFei    0.0      0
1    GuanYu    1.0      1
2         A    NaN      2
3         B    NaN      3
4         C    NaN      4
"""
print(df3)

# 5. outer 外连接
# 外连接相当于求两个 DataFrame 的并集。
df3 = pd.merge(df1, df2, how='outer')
"""
       name  data1  data2
0  ZhangFei    0.0    0.0
1    GuanYu    1.0    1.0
2         a    2.0    NaN
3         b    3.0    NaN
4         c    4.0    NaN
5         A    NaN    2.0
6         B    NaN    3.0
7         C    NaN    4.0
"""
print(df3)

"""
在 Python 里可以直接使用 SQL 语句来操作 Pandas:pandasql
pandasql 中的主要函数是 sqldf，它接收两个参数：
    一个 SQL 查询语句，还有一组环境变量 globals() 或 locals()。
    
这样我们就可以在 Python 里，直接用 SQL 语句中对 DataFrame 进行操作
"""

from pandas import DataFrame
from pandasql import sqldf, load_meat, load_births



df1 = DataFrame({'name':['ZhangFei', 'GuanYu', 'a', 'b', 'c'], 'data1':range(5)})
pysqldf = lambda sql: sqldf(sql, globals())
sql = "select * from df1 where name ='ZhangFei'"
"""
       name  data1
0  ZhangFei      0
"""
print(pysqldf(sql))





