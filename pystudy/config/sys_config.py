#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
@File : sys_config.py
@Author : jeffsheng
@Date : 2019/11/4
@Desc : 
"""
import argparse

parser = argparse.ArgumentParser(description='系统配置')

parser.add_argument('--env', type=str, default='dev', help='tell you the env')

args = parser.parse_args()

