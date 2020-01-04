"""
Get
    urllib的request模块可以非常方便地抓取URL内容，也就是发送一个GET请求到指定的页面，然后返回HTTP的响应
"""
from urllib.parse import urlparse

import urllib

url = 'https://api.douban.com/v2/book/2129650';
o = urlparse(url)
print(o)
# ParseResult(scheme='https', netloc='api.douban.com', path='/v2/book/2129650', params='', query='', fragment='')
print(type(o))
print(o.scheme)  # https

#
# from urllib import request
#
# with request.urlopen('https://api.douban.com/v2/book/2129650') as f:
#     data = f.read()
#     print('Status:', f.status, f.reason)
#     for k, v in f.getheaders():
#         print('%s: %s' % (k, v))
#     print('Data:', data.decode('utf-8'))

