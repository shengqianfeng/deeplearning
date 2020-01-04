#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
@File : data_cleaning.py
@Author : jeffsheng
@Date : 2019/11/7
@Desc : 数据清洗：完全合一
-完整性
-全面性
-合法性
-唯一性
"""
import pandas as pd
from pandas import Series, DataFrame

student = DataFrame(pd.read_excel('data.xlsx'))
"""
  name   age sex
0   张三  25.0   男
1   李四  35.0   女
2   王五  45.0   男
3   赵六   NaN   男
"""
print(student)


# 想对 df[‘Age’] 中缺失的数值用平均年龄进行填充

# student['age'].fillna(student['age'].mean(), inplace=True)
"""
  name   age sex
0   张三  25.0   男
1   李四  35.0   女
2   王五  45.0   男
3   赵六  35.0   男
"""
# print(student)

"""
用最高频的数据进行填充，可以先通过 value_counts 获取 age 字段最高频次 age_maxf，
然后再对 age 字段中缺失的数据用 age_maxf 进行填充
"""
# age_maxf = student['age'].value_counts().index[0]
# student['age'].fillna(age_maxf, inplace=True)
"""
原始数据
 name   age sex
0   张三  25.0   男
1   李四  35.0   女
2   王五  45.0   男
3   赵六   NaN   男
4   田七  25.0   男

填充数据
  name   age sex
0   张三  25.0   男
1   李四  35.0   女
2   王五  45.0   男
3   赵六  25.0   男
4   田七  25.0   男
"""
# print(student)

# 删除全空的行
student.dropna(how='all',inplace=True)
"""
原始数据
name   age  sex
0   张三  25.0    男
1   李四  35.0    女
2  NaN   NaN  NaN
3   王五  45.0    男
4   赵六   NaN    男
5   田七  25.0    男

删除第二行的空行后
  name   age sex
0   张三  25.0   男
1   李四  35.0   女
3   王五  45.0   男
4   赵六   NaN   男
5   田七  25.0   男
"""
print(student)


# 获取 weight 数据列中单位为 lbs 的数据
rows_with_lbs = student['weight'].str.contains('lbs').fillna(False)
"""
原始数据
  name   age sex weight
0   张三  25.0   男  50kgs
1   李四  35.0   女  55lbs
3   王五  45.0   男  57kgs
4   赵六   NaN   男  58lbs
5   田七  25.0   男  59kgs

只列出单位是lbs的
  name   age sex weight
1   李四  35.0   女  55lbs
4   赵六   NaN   男  58lbs
"""
print(student[rows_with_lbs])
print("------------------------")
# 将 lbs转换为 kgs, 2.2lbs=1kgs
for i,lbs_row in student[rows_with_lbs].iterrows():
  # 截取从头开始到倒数第三个字符之前，即去掉lbs。
  weight = int(float(lbs_row['weight'][:-3])/2.2)
  student.at[i,'weight'] = '{}kgs'.format(weight)

"""
全部转化为kgs
  name   age sex weight
0   张三  25.0   男  50kgs
1   李四  35.0   女  24kgs
3   王五  45.0   男  57kgs
4   赵六   NaN   男  26kgs
5   田七  25.0   男  59kgs
"""
print(student)
print("--------------------------------")


# 删除非 ASCII 字符
# student['name'].replace({r'[^\x00-\x7F]+':''}, regex=True, inplace=True)
# print(student)


# 切分名字，删除源数据列
student[['first_name','last_name']] = student['englishname'].str.split(expand=True)
student.drop('name', axis=1, inplace=True)
"""
将englishname切分为两个列
  englishname   age sex weight first_name last_name
0   san zhang  25.0   男  50kgs        san     zhang
1       si Li  35.0   女  24kgs         si        Li
3     wu wang  45.0   男  57kgs         wu      wang
4    liu zhao   NaN   男  26kgs        liu      zhao
5     qi tian  25.0   男  59kgs         qi      tian
6     qi tian  25.0   男  59kgs         qi      tian
"""
print(student)
print("-----------------------------------")

"""
删除重复数据
"""

# 删除重复数据行
student.drop_duplicates(['first_name','last_name'],inplace=True)
"""
  englishname   age sex weight first_name last_name
0   san zhang  25.0   男  50kgs        san     zhang
1       si Li  35.0   女  24kgs         si        Li
3     wu wang  45.0   男  57kgs         wu      wang
4    liu zhao   NaN   男  26kgs        liu      zhao
5     qi tian  25.0   男  59kgs         qi      tian
"""
print(student)







