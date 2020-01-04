#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
@File : server.py
@Author : jeffsheng
@Date : 2019/11/4
@Desc : 
"""
import grpc
import time
from concurrent import futures
from pystudy.grpc.helloworld import data_pb2, data_pb2_grpc

_ONE_DAY_IN_SECONDS = 60 * 60 * 24
_HOST = 'localhost'
_PORT = '50052'

class Greeter(data_pb2_grpc.FormatDataServicer):
    def SayHello(self, request, context):
        str = request.text
        print("server received message:",str)
        return data_pb2.Data(text=str.upper())

    def SaySmile(self, request, context):
        str = request.text
        print("server received message:",str)
        return data_pb2.Data(text=str.upper())

def server():
    grpcServer = grpc.server(futures.ThreadPoolExecutor(max_workers=4))
    data_pb2_grpc.add_FormatDataServicer_to_server(Greeter(), grpcServer)
    grpcServer.add_insecure_port(_HOST + ':' + _PORT)
    grpcServer.start()
    try:
        while True:
            print("server started")
            time.sleep(_ONE_DAY_IN_SECONDS)

    except KeyboardInterrupt:
        grpcServer.stop(0)

if __name__ == '__main__':
    server()


