# Generated by the gRPC Python protocol compiler plugin. DO NOT EDIT!
import grpc

import data_pb2 as data__pb2

# FormatDataStub为客户端相关定义
class FormatDataStub(object):
  # missing associated documentation comment in .proto file
  pass

  def __init__(self, channel):
    """Constructor.

    Args:
      channel: A grpc.Channel.
    """
    self.SaySmile = channel.unary_unary(
        '/jeffsmile.Greeter/SaySmile',# 这路径跟实际包路径无关，跟服务端自定义的接口有关
        request_serializer=data__pb2.Data.SerializeToString,
        response_deserializer=data__pb2.Data.FromString,
        )
#==================== 以上为client.py相关的定义FormatDataStub类========================================================

#==================== 以下为server.py相关的定义FormatDataServicer类和add_FormatDataServicer_to_server方法========================================================
class FormatDataServicer(object):
  # missing associated documentation comment in .proto file
  pass

  def SaySmile(self, request, context):
    # missing associated documentation comment in .proto file
    pass
    context.set_code(grpc.StatusCode.UNIMPLEMENTED)
    context.set_details('Method not implemented!')
    raise NotImplementedError('Method not implemented!')


def add_FormatDataServicer_to_server(servicer, server):
  rpc_method_handlers = {
      'SaySmile': grpc.unary_unary_rpc_method_handler(
          servicer.SaySmile,
          request_deserializer=data__pb2.Data.FromString,
          response_serializer=data__pb2.Data.SerializeToString,
      ),
  }
  generic_handler = grpc.method_handlers_generic_handler(
      'jeffsmile.Greeter', rpc_method_handlers)# # 这路径跟实际包路径无关，自定义接口路径
  server.add_generic_rpc_handlers((generic_handler,))