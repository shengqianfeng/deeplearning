#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
@File : toutiao_spider.py
@Author : jeffsheng
@Date : 2020/2/7 0007
@Desc : 
"""


from random import choice
import requests
from bs4 import BeautifulSoup

user_agent = [
    "Mozilla/5.0 (Macintosh; U; Intel Mac OS X 10_6_8; en-us) AppleWebKit/534.50 (KHTML, like Gecko) Version/5.1 Safari/534.50",
    "Mozilla/5.0 (Windows; U; Windows NT 6.1; en-us) AppleWebKit/534.50 (KHTML, like Gecko) Version/5.1 Safari/534.50",
    "Mozilla/5.0 (Macintosh; Intel Mac OS X 10.6; rv,2.0.1) Gecko/20100101 Firefox/4.0.1",
    "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_7_0) AppleWebKit/535.11 (KHTML, like Gecko) Chrome/17.0.963.56 Safari/535.11",
    "Mozilla/4.0 (compatible; MSIE 7.0; Windows NT 5.1; Maxthon 2.0)",
    "Mozilla/4.0 (compatible; MSIE 7.0; Windows NT 5.1; TencentTraveler 4.0)",
    "Mozilla/4.0 (compatible; MSIE 7.0; Windows NT 5.1; The World)",
    "Mozilla/4.0 (compatible; MSIE 7.0; Windows NT 5.1; Trident/4.0; SE 2.X MetaSr 1.0; SE 2.X MetaSr 1.0; .NET CLR 2.0.50727; SE 2.X MetaSr 1.0)",
    "Mozilla/4.0 (compatible; MSIE 7.0; Windows NT 5.1; 360SE)",
    "Mozilla/4.0 (compatible; MSIE 6.0; Windows NT 5.1; SV1; AcooBrowser; .NET CLR 1.1.4322; .NET CLR 2.0.50727)",
    "Mozilla/4.0 (compatible; MSIE 7.0; Windows NT 6.0; Acoo Browser; SLCC1; .NET CLR 2.0.50727; Media Center PC 5.0; .NET CLR 3.0.04506)",
    "Mozilla/4.0 (compatible; MSIE 7.0; AOL 9.5; AOLBuild 4337.35; Windows NT 5.1; .NET CLR 1.1.4322; .NET CLR 2.0.50727)",
    "Mozilla/5.0 (Windows; U; MSIE 9.0; Windows NT 9.0; en-US)",
    "Mozilla/5.0 (compatible; MSIE 9.0; Windows NT 6.1; Win64; x64; Trident/5.0; .NET CLR 3.5.30729; .NET CLR 3.0.30729; .NET CLR 2.0.50727; Media Center PC 6.0)",
    "Mozilla/5.0 (compatible; MSIE 8.0; Windows NT 6.0; Trident/4.0; WOW64; Trident/4.0; SLCC2; .NET CLR 2.0.50727; .NET CLR 3.5.30729; .NET CLR 3.0.30729; .NET CLR 1.0.3705; .NET CLR 1.1.4322)",
    "Mozilla/4.0 (compatible; MSIE 7.0b; Windows NT 5.2; .NET CLR 1.1.4322; .NET CLR 2.0.50727; InfoPath.2; .NET CLR 3.0.04506.30)",
    "Mozilla/5.0 (Windows; U; Windows NT 5.1; zh-CN) AppleWebKit/523.15 (KHTML, like Gecko, Safari/419.3) Arora/0.3 (Change: 287 c9dfb30)",
    "Mozilla/5.0 (X11; U; Linux; en-US) AppleWebKit/527+ (KHTML, like Gecko, Safari/419.3) Arora/0.6",
    "Mozilla/5.0 (Windows; U; Windows NT 5.1; en-US; rv:1.8.1.2pre) Gecko/20070215 K-Ninja/2.1.1",
    "Mozilla/5.0 (Windows; U; Windows NT 5.1; zh-CN; rv:1.9) Gecko/20080705 Firefox/3.0 Kapiko/3.0",
    "Mozilla/5.0 (X11; Linux i686; U;) Gecko/20070322 Kazehakase/0.4.5",
    "Mozilla/5.0 (X11; U; Linux i686; en-US; rv:1.9.0.8) Gecko Fedora/1.9.0.8-1.fc10 Kazehakase/0.5.6",
    "Mozilla/5.0 (Windows NT 6.1; WOW64) AppleWebKit/535.11 (KHTML, like Gecko) Chrome/17.0.963.56 Safari/535.11",
    "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_7_3) AppleWebKit/535.20 (KHTML, like Gecko) Chrome/19.0.1036.7 Safari/535.20",
    "Mozilla/5.0 (Windows NT 6.1; Win64; x64; rv:56.0) Gecko/20100101 Firefox/56.0",
    "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/76.0.3809.100 Safari/537.36",
]
import re
from pystudy.tools.news.news import News
import json
def get_news_list(url):
    headers = {
        "User-agent": choice(user_agent),
        "cookie":'ttcid=e3a0c13061b0456ca4446e13a3045a1f36; __tasessionId=jmjsnszss1583804903965; csrftoken=f7f56fe7470bbb7ea61c0a76f3d6ccec; tt_webid=6802390218688284167; sso_auth_status=2e7a28fe4bc2b905d820691d76d4221e; sso_uid_tt=4b3975f50eb6c43ea8b4ce2eb4aeda2c; sso_uid_tt_ss=4b3975f50eb6c43ea8b4ce2eb4aeda2c; toutiao_sso_user=37466dadff4fc5cb16f38f8a553215d2; toutiao_sso_user_ss=37466dadff4fc5cb16f38f8a553215d2; tt_webid=6802390218688284167; s_v_web_id=verify_k7l9k17o_15KGdd6T_U6r0_4arW_BQwZ_Lis8cNKxLD2t; SLARDAR_WEB_ID=222b14a3-93b9-4224-a1f6-4966dc04c883; WEATHER_CITY=%E5%8C%97%E4%BA%AC; passport_auth_status=026adf7e3cbef0c74fcba1d8579dd0f2%2C48d95e2607f0f9f1ac4fbdeb6c31ec9e; sid_guard=e529557b3ff65a52275c044ee3eb588b%7C1583808097%7C5184000%7CSat%2C+09-May-2020+02%3A41%3A37+GMT; uid_tt=350bdc4286541c8c91f70577dce3b205; uid_tt_ss=350bdc4286541c8c91f70577dce3b205; sid_tt=e529557b3ff65a52275c044ee3eb588b; sessionid=e529557b3ff65a52275c044ee3eb588b; sessionid_ss=e529557b3ff65a52275c044ee3eb588b; tt_scid=ocK0VpefNrFnH.5ETJcjYm1xoz97YZ85X3tTFH27eHQ4-myDPDyaiea-sumzIQR.91c2'
    }
    data = requests.get(url, headers=headers)
    news_list =  get_toutiao_data(data)
    return news_list

import datetime
today = datetime.datetime.now().strftime('%Y-%m-%d')
def get_toutiao_data(data):
    resp_json = json.loads(data.text)
    news_list = []
    if resp_json['data']==None:
        return news_list
    for dtnew in resp_json['data']:
        if 'merge_article' in dtnew.keys() and  'article_url' not in dtnew.keys():
            for merge_article in dtnew['merge_article']:
                if merge_article['datetime'][0:10] != today:
                    continue
                news = News(merge_article['datetime'], merge_article['title'], merge_article['abstract'], merge_article['article_url'], '今日头条', '疫情谣言',
                            merge_article['image_url'])
                news_list.append(news)
        else:
            if 'datetime' not in dtnew.keys():
                print(dtnew)
                continue
            if dtnew['datetime'][0:10]!=today:
                continue
            summary = ''
            if 'summary' not in dtnew.keys():
                summary =  dtnew['abstract']
            else:
                summary = dtnew['summary']
            news = News(dtnew['datetime'], dtnew['title'], summary, dtnew['article_url'],'今日头条','疫情辟谣',dtnew['image_url'])
            news_list.append(news)
    return news_list


import xlwt

#  将数据写入新文件
def data_write(file_path, datas):
    f = xlwt.Workbook()
    styleBoldRed = xlwt.easyxf('font: color-index red, bold on');
    headerStyle = styleBoldRed

    sheet1 = f.add_sheet(u'sheet1', cell_overwrite_ok=True)  # 创建sheet
    sheet1.write(0, 0, '分类', headerStyle)
    sheet1.write(0, 1, '发布时间', headerStyle)
    sheet1.write(0, 2, '标题', headerStyle)
    sheet1.write(0, 3, '摘要', headerStyle)
    sheet1.write(0, 4, '正文地址', headerStyle)
    sheet1.write(0, 5, '来源', headerStyle)
    sheet1.write(0, 6, '图片地址', headerStyle)
    # 将数据写入第 i 行，第 j 列
    i = 1
    for data in datas:
        sheet1.write(i, 0, data.type)
        sheet1.write(i, 1, data.date)
        sheet1.write(i, 2, data.title)
        sheet1.write(i, 3, data.summary)
        sheet1.write(i, 4, data.content_url)
        sheet1.write(i, 5, data.source)
        sheet1.write(i, 6, data.pic_url)
        i = i + 1

    f.save(file_path)  # 保存文件
    print("--------保存完成！")

if __name__ == '__main__':

    url_indexs=list(range(0,200,20))
    all_list  = []
    for index in url_indexs:
        url = 'https://www.toutiao.com/api/search/content/?aid=24&app_name=web_search&offset=' + str(index) + '&format=json&keyword=%E7%96%AB%E6%83%85%E8%B0%A3%E8%A8%80&autoload=true&count=20&en_qc=1&cur_tab=1&from=search_tab&pd=synthesis&timestamp=1583812117813&_signature=5pEm5AAgEBC1wpJyopXGyuaQZ.AALkBObtNczQ7WF2igUN8.faAoPwQ-DsrmLSDcJTIUX2s5XGhIIc6wTua4KcBHWjQaIOdXmaF6EdlpYAzhlpdZGsiqwUa0JH..LYyY0sL'
        news_list =  get_news_list(url)
        all_list.extend(news_list)
    data_write("./今日头条_疫情谣言"+today+".xls",all_list)
