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

from datetime import datetime, timedelta
def format_date(old_time):
  # time_b = '2020-03-17T04:32:55-04:00'
  # print(old_time[0:19])
  # print(old_time[20:])
  if 'Z' not in old_time:
    utc_time = datetime.strptime(old_time[0:19] + "Z", '%Y-%m-%dT%H:%M:%SZ')
    local_time = utc_time - timedelta(hours=int(old_time[20:][0:2]))
    utc_string = local_time.strftime('%Y-%m-%d %H:%M:%S')
    return utc_string
  else:
    utc_time = datetime.strptime(old_time, '%Y-%m-%dT%H:%M:%S.%fZ')
    local_time = utc_time + timedelta(hours=8)
    utc_string = local_time.strftime('%Y-%m-%d %H:%M:%S')
    return utc_string

df = DataFrame(pd.read_excel('FOX.xls'))

for i in df.index.values:
  print(df.loc[i][1]+"----------"+format_date(df.loc[i][1]))
  df.at[i,'发布时间'] = format_date(df.loc[i][1])

DataFrame(df).to_excel('FOX新闻2020-03-16-1.xlsx', sheet_name='Sheet1', index=False, header=True)