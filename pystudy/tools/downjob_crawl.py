# #!/usr/bin/env python3
# # -*- coding: utf-8 -*-
#
# """
# @File : downjob_crawl.py
# @Author : jeffsheng
# @Date : 2019/11/1
# @Desc :
# """
#
# from urllib.parse import urlencode as UP
# import requests, time, pandas as pd
#
#
# def find_real_url(param):
#     searchs = UP.urlencode(param)
#     url = 'https://www.lagou.com/jobs/list_' + searchs[4:10] + '?px=default&city=%E5%85%A8%E5%9B%BD'
#     return url
#
# def get_json(url,url_json,page,param):  #,page
#     # 请求头
#     headers ={
#         'Host': 'www.lagou.com',
#         'Connection': 'keep-alive',
#         'Origin': 'https://www.lagou.com',
#         'User-Agent':str(us.random),
#         'Content-Type': 'application/x-www-form-urlencoded; charset=UTF-8',
#         'Accept': 'application/json, text/javascript, */*; q=0.01',
#         'X-Requested-With': 'XMLHttpRequest',
#         'Referer': str(url),
#         'Accept-Language': 'zh-CN,zh;q=0.9'
#     }
#     # 抓包得到必须要传递的data参数
#     data = {
#         'first':'false',
#         'pn'    :page,
#         'kd'    :param["job"]
#     }
#     sess = requests.session()   #创建一个session对象
#     sess.headers.update(headers)
#     try:
#
#         html1 = sess.get(url,headers= headers)    #先get
#     except:
#         retry_count = 5
#         while retry_count > 0:
#             proxy = get_proxy()
#             try:
#                 html1 = sess.get(url,headers= headers,proxies={"http": "http://{}".format(proxy)})    #先get
#             except Exception:
#                 print("retry_count="+retry_count,url)
#                 retry_count -= 1
#                 delete_proxy(proxy)
#     # 出错5次, 删除代理池中代理
#     if html1.status_code==200:   #判断请求后的状态码是否请求成功
#         time.sleep(1)
#         try:
#             html = sess.post(url=url_json, headers=headers,data=data)
#             html.raise_for_status()    # post请求不成功，则引发HTTPError异常
#             return html.json()
#         except:
#             print(str(page)+'th页出现异常')
#
#
# def save(total,job):
#     # 保存至csv文件
#     table_Lables = ['公司ID', '公司名称', '公司类型', '发布时间','是否要求有经验','学历要求' ,'工作类型', '薪资', '公司大小', '职位名称', '所在城市', '技能标签',
#                     '福利待遇']
#     data_list = []
#     for data_json in total:
#         for i in data_json['content']['positionResult']['result']:
#             try:
#                 data_list.append([i['companyId'], i['companyFullName'], i['industryField'], i['formatCreateTime'],i['workYear'],
#                               i['education'],i['jobNature'], i['salary'], i['companySize'], i['positionName'], i['city'],
#                               str(i['skillLables']),i['positionAdvantage']])
#
#             except Exception as inst:
#                 print(inst.args)
#                 continue
#
#     data_write = pd.DataFrame(columns=table_Lables, data=data_list)
#     data_write.to_csv('data/%s.csv'%job, index=False, encoding='utf_8_sig')
#
#
# def crawl(param):
#     url_html = find_real_url(param)
#     total = []
#     url_json = 'https://www.lagou.com/jobs/positionAjax.json?&px=default&needAddtionalResult=false'
#     for page in range(1,pages+1):
#         try:
#             data_json = get_json(url_html,url_json, page,param)
#             total.append(data_json)
#             print('正在采集'+param["job"]+'第%d页'%page)
#         except Exception as inst:
#             print(page,"页异常",inst.args)
#             continue
#     save(total,job)
#
# if __name__ == '__main__':
#    jobs = ["PHP","JAVA","数据挖掘","搜索算法","精准推荐","C","C#","全栈工程师",".NET","Hadoop","Python","Delphi","VB","Perl","Ruby","Node.js","Go","ASP","Shell","区块链","后端开发其它HTML5","Android","iOS","WP","web前端","Flash","html5","JavaScript","U3D","COCOS2D-X","深度学习","机器学习","图像处理","图像识别","语音识别","机器视觉","算法工程师","自然语言处理"]
#
#    for job in jobs:
#         city=""
#         pages = 100
#         search = {
#             'job':job
#         }
#         data_list = []
#         try:
#            crawl(param)
#         except Exception as inst:
#             print(job,inst.args)
#             continue
#
