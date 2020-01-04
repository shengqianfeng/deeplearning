import requests

r = requests.get('https://www.douban.com/')  # 豆瓣首页

print(r.status_code)    # 200

# print(r.text)

# get带参数
r = requests.get('https://www.douban.com/search', params={'q': 'python', 'cat': '1001'})
print(r.url)  # https://www.douban.com/search?q=python&cat=1001
print(r.encoding)   # utf-8
print( r.content)


# 获取json结果
# r = requests.get('https://query.yahooapis.com/v1/public/yql?q=select%20*%20from%20weather.forecast%20where%20woeid%20%3D%202151330&format=json')
# print(r.json())


# 传入HTTP Header
# r = requests.get('https://www.douban.com/', headers={'User-Agent': 'Mozilla/5.0 (iPhone; CPU iPhone OS 11_0 like Mac OS X) AppleWebKit'})
# print(r.text)



# post + data
# r = requests.post('https://accounts.douban.com/login', data={'form_email': 'abc@example.com', 'form_password': '123456'})
# print(r.text)

"""
post+json参数：
params = {'key': 'value'}
r = requests.post(url, json=params) # 内部自动序列化为JSON
"""


"""
上传文件：files
    在读取文件时，注意务必使用'rb'即二进制模式读取，这样获取的bytes长度才是文件的长度
"""
# upload_files = {'file': open('report.xls', 'rb')}
# r = requests.post(url, files=upload_files)


# 把post()方法替换为put()，delete()等，就可以以PUT或DELETE方式请求资源。
# 获取响应头
print(r.headers)


"""
轻松获取指定cookie
"""

# print(r.cookies['ts'])


"""
传入cookie
"""
# cs = {'token': '12345', 'status': 'working'}
# r = requests.get(url, cookies=cs)

# 要指定超时，传入以秒为单位的timeout参数：
# r = requests.get(url, timeout=2.5)  # 2.5秒后超时












