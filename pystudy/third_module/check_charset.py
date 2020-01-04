"""
chardet
    检测编码
"""
import chardet

# 检测一个bytes的编码 confidence字段，表示检测的概率是1.0（即100%）
a = chardet.detect(b'Hello, world!')
print(a)    # {'encoding': 'ascii', 'confidence': 1.0, 'language': ''}


# 检测gbk的编码
data = '离离原上草，一岁一枯荣'.encode('gbk')
print(chardet.detect(data))    # {'encoding': 'GB2312', 'confidence': 0.7407407407407407, 'language': 'Chinese'}


# 检测utf8的编码
data = '离离原上草，一岁一枯荣'.encode('utf-8')
print(chardet.detect(data)) # {'encoding': 'utf-8', 'confidence': 0.99, 'language': ''}

# 检测日文编码
data = '最新の主要ニュース'.encode('euc-jp')
print(chardet.detect(data))     # {'encoding': 'EUC-JP', 'confidence': 0.99, 'language': 'Japanese'}







