#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
@File : remote.py
@Author : jeffsheng
@Date : 2019/12/16
@Desc : 
"""
import requests
import urllib
from datetime import date
import json
from bs4 import BeautifulSoup
import webbrowser
import re

# 403
# r = requests.get('http://m.maoyan.com/ajax/detailmovie?movieId=1238837')
# print(r.content)
# 403
# r = requests.get('http://m.maoyan.com/ajax/movieOnInfoList')
# print(r.content)

# url = 'http://m.maoyan.com/ajax/movieOnInfoList'
# print("---------------------get请求----------------------")
# url='http://m.maoyan.com/ajax/detailmovie?movieId=1238837'


headers = {"User-Agent": "Mozilla/5.0 (Windows NT 10.0; WOW64)",
           'Cookie': 'uuid_n_v=v1; iuuid=8134CA701CCF11EAA13BE1E4D7BDBC441D118652C89D487390980C79FF1DEFDD; webp=true; ci=1%2C%E5%8C%97%E4%BA%AC',}
# request = urllib.request.Request(url=url, headers=headers)
# response = urllib.request.urlopen(request)
# content = response.read().decode('utf-8')
# print(content)
print("---------------------带参数的get请求----------------------")
url = 'http://m.maoyan.com/ajax/detailmovie?movieId=1238837'
# url = 'http://m.maoyan.com/cinemas.json'
# postData ={'cityId': '30', 'limit': 12, 'movieId': '1238837', 'day': '2019-12-17', 'offset': 144}
request = urllib.request.Request(url=url, headers=headers)
response = urllib.request.urlopen(request).read().decode('utf-8')
print(response)


# reqCode = '6ac3ab21b18745a1a77b2b4299530f2b'
# url = 'https://verify.meituan.com/v2/web/general_page?action=spiderindefence&requestCode=' + reqCode + '&platform=1000&adaptor=auto&succCallbackUrl=https://optimus-mtsi.meituan.com/optimus/verifyResult?originUrl=http%3A%2F%2Fm.maoyan.com%2Fajax%2Fdetailmovie%3FmovieId%3D1238837'
