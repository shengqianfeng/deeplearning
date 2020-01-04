#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
@File : load_conf.py
@Author : jeffsheng
@Date : 2019/11/4
@Desc : 
"""
from configobj import ConfigObj
from pystudy.config import sys_config
# 加载参数
args = sys_config.args

class LoadConf(object):
    def __init__(self):
        # 读取配置环境
        print("读取配置环境_初始化")
        self.config = ConfigObj("../config/conf.ini", encoding='UTF8')
        if not self.config:
            self.config = ConfigObj("config/conf.ini", encoding='UTF8')
        # 默认开发环境数据
        self.conf_dict = self.config[args.env]
        print("loadconf is over！")

    def set_conf(self, type):
        self.conf_dict = self.config[type]
        return self.conf_dict

    def get_conf(self):
        return self.conf_dict

if __name__ == '__main__':
    LoadConf()

