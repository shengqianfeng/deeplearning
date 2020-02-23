#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
@File : news.py
@Author : jeffsheng
@Date : 2020/2/7 0007
@Desc : 
"""
import json

class News(object):

    def __init__(self, date,title,summary,content_url,source,type,pic_url):
        self.date = date.strip()
        self.title  = title.strip()
        self.summary = summary.strip()
        self.content_url = content_url.strip()
        self.source = source.strip()
        self.type = type.strip()
        self.pic_url = pic_url.strip()

    def __str__(self):
        return json.dumps(self.__dict__, ensure_ascii=False)




if __name__ == '__main__':
    new = News("1","你好","3","4")
    print(new)