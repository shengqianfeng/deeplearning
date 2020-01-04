#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
@File : down_wangyi.py
@Author : jeffsheng
@Date : 2019/11/1
@Desc : 根据关键词下载网易严选的商品图片
"""

import requests,os


def search_keyword(keyword):
    uri = 'https://you.163.com/xhr/search/search.json'
    query = {
        "keyword": keyword,
        "page": 1
    }

    try:
        res = requests.get(uri, params=query).json()
        result = res['data']['directly']['searcherResult']['result']
        print(result)
        i=1
        road = "D://download//" + keyword
        if not os.path.exists(road):
            os.mkdir(road)
        for r in result:
           url = r['primaryPicUrl'];
           try:
               pic = requests.get(url, timeout=5)  # 可能有些图片存在网址打不开的情况,这里设置一个5秒的超时控制
           except Exception:  # 出现异常直接跳过
               print("【错误】当前图片无法下载")
               continue  # 跳过本次循环
           #  定义变量保存图片的路径
           fp = open('D://download//' + keyword + "//" + str(i) + ".jpg", 'wb')
           fp.write(pic.content)
           fp.close()
           i+=1
    except:
        raise


if __name__ == '__main__':
    result = search_keyword("裤子")
    print(result)