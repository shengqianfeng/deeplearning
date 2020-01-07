# -*- coding: utf-8 -*-
"""
日志初始化--grpc 子目录启动使用路径
"""

from pystudy.sysutils import log_utils
import os
import time

# 子路径启动使用日志路径
log_path = "../logs"


class LogConf(object):
    def __init__(self):
        print("日志初始化")
        # 默认日志地址
        self.log_manage = log_utils.LogManage()
        # if not os.path.exists(log_path):
        #     os.makedirs(log_path)
        # 日志备份
        log_new_path = os.path.join(log_path, "log")
        # if os.path.exists(os.path.join(log_new_path, "log.txt")):
        #     log_back_path = os.path.join(log_path, "log_" + str(time.strftime('%Y%m%d%H%M%S', time.localtime())))
        #     os.system("mv %s %s" % (log_new_path, log_back_path))
        # 创建新日志路径
        if not os.path.exists(log_new_path):
            os.makedirs(log_new_path)
        # 创建日志
        self.log_file_path = os.path.join(log_new_path, "log.txt")
        # self.logger = self.log_manage.get_logger(self.log_file_path, "api_log")

    def get_log(self, name):
        return self.log_manage.get_logger(self.log_file_path, name)


log_base_config = LogConf()
