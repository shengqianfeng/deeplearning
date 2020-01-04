#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
@File : mongodb.py
@Author : jeffsheng
@Date : 2019/11/4
@Desc : 
"""
from pymongo import MongoClient

TABLE_NAME = "student"
student_dict = {}

class DBManager(object):
    def __init__(self):
        print('DBManager init')
        # 连接MongoDB
        self.conn = MongoClient("127.0.0.1:27017")
        # 指定数据库
        self.db = self.conn["jeffsmile"]
        # 指定集合
        student = self.db[TABLE_NAME]
        for entity in student.find():
            student_dict[str(entity['_id'])] = entity

    def getstudent(self):
        print(student_dict)

    """
    db.集合名称.aggregate([{管道:{表达式}}])
        $group：将集合中的文档分组，可用于统计结果
        $match：过滤数据，只输出符合条件的文档
        $project：修改输入文档的结构，如重命名、增加、删除字段、创建计算结果
        $sort：将输入文档排序后输出
        $limit：限制聚合管道返回的文档数
        $skip：跳过指定数量的文档，并返回余下的文档
        $unwind：将数组类型的字段进行拆分
        $strLenCP:计算某个字段的长度
    """
    def aggregate(self):
        stu_list = []
        student_list = self.db[TABLE_NAME].aggregate([{"$match": {"sex": {"$in": ['woman']}}},
                                               {"$project": {"name": 1, "length": {"$strLenCP": "$name"}}},
                                               {"$sort": {"length": -1}}
                                                     ])
        for i in student_list:
            if i is not None:
                print(i["name"])
        return stu_list

if __name__ == '__main__':
    DBManager().getstudent()
    stu_list = DBManager().aggregate()
