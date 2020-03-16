#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
@File : regular.py
@Author : jeffsheng
@Date : 2019/11/5
@Desc : 正则表达式 Python提供re模块，包含所有正则表达式的功能
"""

"""
\d可以匹配一个数字
\w可以匹配一个字母或数字
.可以匹配任意字符
*表示任意个字符（包括0个)
+表示至少一个字符
?表示0个或1个字符
{n}表示n个字符
{n,m}表示n-m个字符
[]表示范围
[0-9a-zA-Z\_]可以匹配一个数字、字母或者下划线
[0-9a-zA-Z\_]+可以匹配至少由一个数字、字母或者下划线组成的字符串
[a-zA-Z\_][0-9a-zA-Z\_]*可以匹配由字母或下划线开头，后接任意个由一个数字、字母或者下划线组成的字符串，也就是Python合法的变量
[a-zA-Z\_][0-9a-zA-Z\_]{0, 19}更精确地限制了变量的长度是1-20个字符（前面1个字符+后面最多19个字符）
A|B可以匹配A或B
^表示行的开头，^\d表示必须以数字开头
$表示行的结束，\d$表示必须以数字结束

解读：
    \d{3}表示匹配3个数字
    \s+表示至少有一个空格
    \d{3,8}表示3-8个数字
"""

# 由于Python的字符串本身也用\转义，所以要特别注意
s = 'ABC\\-001'  # Python的字符串   应的正则表达式字符串变成： 'ABC\-001'

# 因此我们强烈建议使用Python的r前缀，就不用考虑转义的问题了
s = r'ABC\-001'  # Python的字符串   对应的正则表达式字符串不变： 'ABC\-001'


import re


"""
match()方法判断是否匹配，如果匹配成功，返回一个Match对象，否则返回None
"""
# <re.Match object; span=(0, 9), match='010-12345'>
print(re.match(r'^\d{3}\-\d{3,8}$', '010-12345'))
print(re.match(r'^\d{3}\-\d{3,8}$', '010 12345'))   # None
# test = '用户输入的字符串'
# if re.match(r'正则表达式', test):
#     print('ok')
# else:
#     print('failed')



"""
切分字符串

用正则表达式切分字符串比用固定的字符更灵活
"""
print('a b   c'.split(' '))
# ['a', 'b', '', '', 'c'] 无法识别连续的空格，需要使用正则表达式无论多少个空格都可以正常分割
print(re.split(r'\s+', 'a b   c'))  # ['a', 'b', 'c']
# 加入,试试 虽然，后有多个空格，但是也可以识别
print(re.split(r'[\s\,]+', 'a,b, c  d'))    # ['a', 'b', 'c', 'd']
# 再加入;试试  分割逗号、空格或者分号都可以
print(re.split(r'[\s\,\;]+', 'a,b;; c  d'))     # ['a', 'b', 'c', 'd']


"""
正则表达式的分组功能
注意到group(0)永远是原始字符串，group(1)、group(2)……表示第1、2、……个子串
"""
m = re.match(r'^(\d{3})-(\d{3,8})$', '010-12345')
print(m.group(0))   # 010-12345
print(m.group(1))   # 010
print(m.group(2))   # 12345

t = '19:05:30'
m = re.match(r'^(0[0-9]|1[0-9]|2[0-3]|[0-9])\:(0[0-9]|1[0-9]|2[0-9]|3[0-9]|4[0-9]|5[0-9]|[0-9])\:(0[0-9]|1[0-9]|2[0-9]|3[0-9]|4[0-9]|5[0-9]|[0-9])$',t)
print(m.group())    # 19:05:30


"""
贪婪匹配

最后需要特别指出的是，正则匹配默认是贪婪匹配，也就是匹配尽可能多的字符
"""
print(re.match(r'^(\d+)(0*)$', '102300').groups())  # ('102300', '')
# 由于\d+采用贪婪匹配，直接把后面的0全部匹配了，结果0*只能匹配空字符串了
# 必须让\d+采用非贪婪匹配（也就是尽可能少匹配），才能把后面的0匹配出来,
# 加个?就可以让\d+采用非贪婪匹配
print(re.match(r'^(\d+?)(0*)$', '102300').groups())     # ('1023', '00')


"""
编译

当我们在Python中使用正则表达式时，re模块内部会干两件事情：

编译正则表达式，如果正则表达式的字符串本身不合法，会报错；

用编译后的正则表达式去匹配字符串。

如果一个正则表达式要重复使用几千次，出于效率的考虑，我们可以预编译该正则表达式，接下来重复使用时就不需要编译这个步骤了，直接匹配
"""
# 编译:
re_telephone = re.compile(r'^(\d{3})-(\d{3,8})$')

# 使用：
print(re_telephone.match('010-12345').groups())
print(re_telephone.match('010-8086').groups())

# 编译后生成Regular Expression对象，由于该对象自己包含了正则表达式，所以调用对应的方法时不用给出正则字符串。


print("-----------------")
list = re.split(r'\s+', '\n\na\n\n\nb\n\n\n   c')
print(list)  # ['a', 'b', 'c']
print([x for x in list if x])
print("----------------------------------------------")
text = '__jp0({"data":{"data":[{"id":"18038568187910963786","recoid":"12877882394207664025","summary":"","title":"意大利一市长为辱华言论道歉：被假新闻蒙蔽了","url":"http:\/\/m.uczzd.cn\/webview\/video?app=quark-iflow&aid=18038568187910963786&cid=10520&zzd_from=quark-iflow&uc_param_str=dndsfrvesvntnwpfgicpbi&recoid=12877882394207664025&rd_type=reco&original_url=http%3A%2F%2Fv.ums.uc.cn%2Fvideo%2Fv_b76847469f2a9258.html&sp_gz=0&uc_biz_str=S%3Acustom%7CC%3Aiflow_video_hide&ums_id=b76847469f2a9258&fallback=true&activity=1&activity2=1","publish_time":1584256883000,"source":"老王带你看新闻","time":"19小时前","pic":["\/\/image.uczzd.cn\/2853248508070961983.jpg?id=0"]},{"id":"17381255073245416159","recoid":"12877882394207664025","summary":"","title":"湖北低风险街道乡镇“解封”","url":"http:\/\/m.uczzd.cn\/webview\/news?app=quark-iflow&aid=17381255073245416159&cid=10520&zzd_from=quark-iflow&uc_param_str=dndsfrvesvntnwpfgibicp&recoid=12877882394207664025&rd_type=reco&sp_gz=0&fallback=true&activity=1&activity2=1","publish_time":1584254340000,"source":"光明网","time":"20小时前","pic":[]},{"id":"2661188416450827399","recoid":"12877882394207664025","summary":"","title":"新冠肺炎表彰名单来了!没有钟南山跟李兰娟,也没有张文宏","url":"http:\/\/m.uczzd.cn\/webview\/video?app=quark-iflow&aid=2661188416450827399&cid=10520&zzd_from=quark-iflow&uc_param_str=dndsfrvesvntnwpfgicpbi&recoid=12877882394207664025&rd_type=reco&original_url=http%3A%2F%2Fv.ums.uc.cn%2Fvideo%2Fv_803b434a96c2532d.html&sp_gz=0&uc_biz_str=S%3Acustom%7CC%3Aiflow_video_hide&ums_id=803b434a96c2532d&fallback=true&activity=1&activity2=1","publish_time":1583724652000,"source":"浙样的生活","time":"6天前","pic":["\/\/image.uczzd.cn\/708703023465382634.jpg?id=0"]},{"id":"3334814591274148252","recoid":"12877882394207664025","summary":"","title":"《纽约时报》这篇文章, 让中国人一声叹息","url":"http:\/\/m.uczzd.cn\/webview\/news?app=quark-iflow&aid=3334814591274148252&cid=10520&zzd_from=quark-iflow&uc_param_str=dndsfrvesvntnwpfgibicp&recoid=12877882394207664025&rd_type=reco&sp_gz=0&fallback=true&activity=1&activity2=1","publish_time":1584261373000,"source":"中国经济网","time":"18小时前","pic":["\/\/image.uczzd.cn\/5756204879655669766.jpg?id=0"]},{"id":"9817543838579951989","recoid":"12877882394207664025","summary":"","title":"世界告急, 中国放大招, 一省救一国, “白衣侠”逆行支援“地球村”","url":"http:\/\/m.uczzd.cn\/webview\/news?app=quark-iflow&aid=9817543838579951989&cid=10520&zzd_from=quark-iflow&uc_param_str=dndsfrvesvntnwpfgibicp&recoid=12877882394207664025&rd_type=reco&sp_gz=0&fallback=true&activity=1&activity2=1","publish_time":1584086071000,"source":"冰视频","time":"2天前","pic":["\/\/image.uczzd.cn\/9790856011701633050.jpg?id=0"]},{"id":"5075435619179194810","recoid":"12877882394207664025","summary":"","title":"清明期间浙江暂停组织群众集中祭扫活动","url":"http:\/\/m.uczzd.cn\/webview\/news?app=quark-iflow&aid=5075435619179194810&cid=10520&zzd_from=quark-iflow&uc_param_str=dndsfrvesvntnwpfgibicp&recoid=12877882394207664025&rd_type=reco&sp_gz=0&fallback=true&activity=1&activity2=1","publish_time":1584259912000,"source":"钱塘新区发布","time":"19小时前","pic":["\/\/image.uczzd.cn\/1641802445141489788.jpg?id=0"]},{"id":"565063882444208378","recoid":"12877882394207664025","summary":"","title":"越南首富千金在英国感染新冠肺炎，父亲包机将其接回并确诊","url":"http:\/\/m.uczzd.cn\/webview\/news?app=quark-iflow&aid=565063882444208378&cid=10520&zzd_from=quark-iflow&uc_param_str=dndsfrvesvntnwpfgibicp&recoid=12877882394207664025&rd_type=reco&sp_gz=0&fallback=true&activity=1&activity2=1","publish_time":1584270577000,"source":"海峡都市报","time":"16小时前","pic":["\/\/image.uczzd.cn\/10540712866717474577.jpg?id=0","\/\/image.uczzd.cn\/14199271567051015491.jpg?id=0","\/\/image.uczzd.cn\/8977995363568177267.jpg?id=0"]},{"id":"3714812977587323657","recoid":"12877882394207664025","summary":"","title":"杭州通报一例无症状感染者! 担心身边有无症状感染者? 看这些重点!","url":"http:\/\/m.uczzd.cn\/webview\/news?app=quark-iflow&aid=3714812977587323657&cid=10520&zzd_from=quark-iflow&uc_param_str=dndsfrvesvntnwpfgibicp&recoid=12877882394207664025&rd_type=reco&sp_gz=0&fallback=true&activity=1&activity2=1","publish_time":1584013395000,"source":"江干发布","time":"3天前","pic":["\/\/image.uczzd.cn\/10850566625665419997.jpg?id=0"]},{"id":"13422373072845316714","recoid":"12877882394207664025","summary":"","title":"湖北男子隔离被收14000元 官方公布真相后网友回应：收太少了","url":"http:\/\/m.uczzd.cn\/webview\/video?app=quark-iflow&aid=13422373072845316714&cid=10520&zzd_from=quark-iflow&uc_param_str=dndsfrvesvntnwpfgicpbi&recoid=12877882394207664025&rd_type=reco&original_url=http%3A%2F%2Fv.ums.uc.cn%2Fvideo%2Fv_455f4a0eabe305bd.html&sp_gz=0&uc_biz_str=S%3Acustom%7CC%3Aiflow_video_hide&ums_id=455f4a0eabe305bd&fallback=true&activity=1&activity2=1","publish_time":1584270580000,"source":"梨视频","time":"16小时前","pic":["\/\/image.uczzd.cn\/3465066578352033768.jpg?id=0","\/\/image.uczzd.cn\/6737838291712826120.jpg?id=0","\/\/image.uczzd.cn\/15975990596998317219.jpg?id=0"]},{"id":"9639580676826357538","recoid":"12877882394207664025","summary":"","title":"那个从美国回国求医的黎女士, 还有更多问题…","url":"http:\/\/m.uczzd.cn\/webview\/news?app=quark-iflow&aid=9639580676826357538&cid=10520&zzd_from=quark-iflow&uc_param_str=dndsfrvesvntnwpfgibicp&recoid=12877882394207664025&rd_type=reco&sp_gz=0&fallback=true&activity=1&activity2=1","publish_time":1584316668000,"source":"环球网","time":"3小时前","pic":["\/\/image.uczzd.cn\/8086000652909230917.jpg?id=0","\/\/image.uczzd.cn\/4744108899578981118.jpg?id=0","\/\/image.uczzd.cn\/12122642799976137273.jpg?id=0"]},{"id":"890711993102958006","recoid":"12877882394207664025","summary":"","title":"不简单! 郑州“毒王”去欧洲疑点重重!","url":"http:\/\/m.uczzd.cn\/webview\/news?app=quark-iflow&aid=890711993102958006&cid=10520&zzd_from=quark-iflow&uc_param_str=dndsfrvesvntnwpfgibicp&recoid=12877882394207664025&rd_type=reco&sp_gz=0&fallback=true&activity=1&activity2=1","publish_time":1584320314000,"source":"薇父私房","time":"2小时前","pic":["\/\/image.uczzd.cn\/5002233340287376512.jpg?id=0","\/\/image.uczzd.cn\/8482904731773687539.jpg?id=0","\/\/image.uczzd.cn\/12175904422748179539.jpg?id=0"]},{"id":"16541929634550032439","recoid":"12877882394207664025","summary":"","title":"“以后每年我都会求一次婚! ”在杭州刷屏的95后美小护结束隔离, 男友又向她求了一次婚","url":"http:\/\/m.uczzd.cn\/webview\/news?app=quark-iflow&aid=16541929634550032439&cid=10520&zzd_from=quark-iflow&uc_param_str=dndsfrvesvntnwpfgibicp&recoid=12877882394207664025&rd_type=reco&sp_gz=0&fallback=true&activity=1&activity2=1","publish_time":1583858694000,"source":"钱江晚报","time":"5天前","pic":["\/\/image.uczzd.cn\/3155440540898228275.jpg?id=0","\/\/image.uczzd.cn\/1876280518040152955.jpg?id=0","\/\/image.uczzd.cn\/4217091377665336384.jpg?id=0"]},{"id":"751032615012661597","recoid":"12877882394207664025","summary":"","title":"特朗普: 未来8周美国疫情可能会变糟 但我不会为此负责","url":"http:\/\/m.uczzd.cn\/webview\/video?app=quark-iflow&aid=751032615012661597&cid=10520&zzd_from=quark-iflow&uc_param_str=dndsfrvesvntnwpfgicpbi&recoid=12877882394207664025&rd_type=reco&original_url=http%3A%2F%2Fv.ums.uc.cn%2Fvideo%2Fv_83614429981f1168.html&sp_gz=0&uc_biz_str=S%3Acustom%7CC%3Aiflow_video_hide&ums_id=83614429981f1168&fallback=true&activity=1&activity2=1","publish_time":1584145569000,"source":"青蜂侠Bee","time":"2天前","pic":["\/\/image.uczzd.cn\/1111878930805942437.jpg?id=0"]}],"location":"深圳"}});'
text = re.sub('^\_\_jp\d{1,3}\(*','',text)
text = re.sub('\)\;$','',text)
print(text)