#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
@File : dxys_spider.py
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

def get_news_list(url):
    headers = {
        "User-agent": choice(user_agent)
    }
    html = requests.get(url, headers=headers)
    news_list = get_jkcenter_data(html)
    return news_list


def get_jkcenter_data(html):
    html.encoding = 'UTF-8'
    soup = BeautifulSoup(html.text, 'html.parser')
    dtnews = soup.find('ul', attrs={"class": "jal-item-list"}).find_all('li')
    news_list = []
    for dtnew in dtnews:
        content_url = 'http://www.chinacdc.cn/jkzt/crb/zl/szkb_11803/jszl_2275/' + dtnew.find('a')['href']
        news_info = [x for x in re.split(r'\n+', dtnew.text) if x]
        news = News(news_info[1], news_info[0], news_info[0], content_url)
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
        sheet1.write(i, 0, '')
        sheet1.write(i, 1, data.date)
        sheet1.write(i, 2, data.title)
        sheet1.write(i, 3, data.summary)
        sheet1.write(i, 4, data.content_url)
        sheet1.write(i, 5, '国家疾控中心')
        sheet1.write(i, 6, '')
        i = i + 1

    f.save(file_path)  # 保存文件
    print("--------保存完成！")

if __name__ == '__main__':
    urls = [
     'http://www.chinacdc.cn/jkzt/crb/zl/szkb_11803/jszl_2275/index.html',
     'http://www.chinacdc.cn/jkzt/crb/zl/szkb_11803/jszl_2275/index_1.html',
     'http://www.chinacdc.cn/jkzt/crb/zl/szkb_11803/jszl_2275/index_2.html',
     'http://www.chinacdc.cn/jkzt/crb/zl/szkb_11803/jszl_2275/index_3.html',
     'http://www.chinacdc.cn/jkzt/crb/zl/szkb_11803/jszl_2275/index_4.html',
     'http://www.chinacdc.cn/jkzt/crb/zl/szkb_11803/jszl_2275/index_5.html',
     'http://www.chinacdc.cn/jkzt/crb/zl/szkb_11803/jszl_2275/index_6.html'
        ]
    all_list  = []
    for url in urls:
        print(url)
        news_list = get_news_list(url)
        all_list.extend(news_list)
    data_write("./国家疾控中心_知识天地.xls",all_list)
