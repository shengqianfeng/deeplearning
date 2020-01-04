#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
@File : client.py
@Author : jeffsheng
@Date : 2019/11/4
@Desc : 
"""

import grpc
from pystudy.grpc.helloworld  import data_pb2, data_pb2_grpc

_HOST = 'localhost'
_PORT = '50052'


def run():
    conn = grpc.insecure_channel(_HOST + ':' + _PORT)
    stub = data_pb2_grpc.FormatDataStub(channel=conn)
    response = stub.SaySmile(data_pb2.Data(text='hello,world!'))
    print("client received: " + response.text)


if __name__ == '__main__':
    run()

