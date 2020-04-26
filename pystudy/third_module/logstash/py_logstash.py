#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
@File : py_logstash.py
@Author : jeffsheng
@Date : 2020/4/16
@Desc : 
"""
import logging
import logstash
import sys

#host为logstash的IP地址
host = '47.107.125.20'

test_logger = logging.getLogger('python-logstash-logger')
test_logger.setLevel(logging.INFO)

#创建一个logHandler
test_logger.addHandler(logstash.LogstashHandler(host, 5959))

test_logger.error('这是一行测试日志')
test_logger.info('python-logstash: test logs  tash info message.')
test_logger.warning('python-logstash: test logstash warning message.')
print("------end-----------")


