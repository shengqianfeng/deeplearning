#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
@File : beautiful_soup.py
@Author : jeffsheng
@Date : 2019/11/1
@Desc :  使用BeautifulSoup插件解析html
"""

from bs4 import BeautifulSoup
import requests

url = 'https://www.zhipin.com/job_detail/?query=python&city=101010100'

res = requests.get(url, headers={}).text
print(res)
content = BeautifulSoup(res, "html.parser")
ul = content.find_all('div')
print(ul)



