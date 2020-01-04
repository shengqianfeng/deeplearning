#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
@File : selenium_webdriver_fox.py
@Author : jeffsheng
@Date : 2019/12/18
@Desc : 根据selenium截取图片
"""
import json
from io import BytesIO

import time
from PIL import Image
from selenium import webdriver
from random import choice
from pystudy.tools.chaojiying_vercode_recognition import get_res
# uuid_n_v=v1; iuuid=8134CA701CCF11EAA13BE1E4D7BDBC441D118652C89D487390980C79FF1DEFDD; webp=true; ci=30%2C%E6%B7%B1%E5%9C%B3

cookies =[
    {
        "name": "ci",
        "value": "30%2C%E6%B7%B1%E5%9C%B3",
    },
    {

        "name": "iuuid",
        "value": "8134CA701CCF11EAA13BE1E4D7BDBC441D118652C89D487390980C79FF1DEFDD",
    },
    {

        "name": "uuid_n_v",
        "value": "v1",
    },
    {

        "name": "webp",
        "value": "true",
    }
]

user_agent = [
    "Mozilla/5.0 (Macintosh; U; Intel Mac OS X 10_6_8; en-us) AppleWebKit/534.50 (KHTML, like Gecko) Version/5.1 Safari/534.50",
    "Mozilla/5.0 (Windows; U; Windows NT 6.1; en-us) AppleWebKit/534.50 (KHTML, like Gecko) Version/5.1 Safari/534.50",
    "Mozilla/5.0 (Macintosh; Intel Mac OS X 10.6; rv,2.0.1) Gecko/20100101 Firefox/4.0.1",
    "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_7_0) AppleWebKit/535.11 (KHTML, like Gecko) Chrome/17.0.963.56 Safari/535.11",
    "Mozilla/4.0 (compatible; MSIE 7.0; Windows NT 5.1; Maxthon 2.0)",
    "Mozilla/4.0 (compatible; MSIE 7.0; Windows NT 5.1; TencentTraveler 4.0)",
    "Mozilla/4.0 (compatible; MSIE 7.0; Windows NT 5.1; The World)",
    "Mozilla/4.0 (compatible; MSIE 7.0; Windows NT 5.1; Trident/4.0; SE 2.X MetaSr 1.0; SE 2.X MetaSr 1.0; .NET CLR 2.0.50727; SE 2.X MetaSr 1.0)",
    "Mozilla/4.0 (compatible; MSIE 7.0; Windows NT 5.1; 360SE)",
    "Mozilla/4.0 (compatible; MSIE 6.0; Windows NT 5.1; SV1; AcooBrowser; .NET CLR 1.1.4322; .NET CLR 2.0.50727)",
    "Mozilla/4.0 (compatible; MSIE 7.0; Windows NT 6.0; Acoo Browser; SLCC1; .NET CLR 2.0.50727; Media Center PC 5.0; .NET CLR 3.0.04506)",
    "Mozilla/4.0 (compatible; MSIE 7.0; AOL 9.5; AOLBuild 4337.35; Windows NT 5.1; .NET CLR 1.1.4322; .NET CLR 2.0.50727)",
    "Mozilla/5.0 (Windows; U; MSIE 9.0; Windows NT 9.0; en-US)",
    "Mozilla/5.0 (compatible; MSIE 9.0; Windows NT 6.1; Win64; x64; Trident/5.0; .NET CLR 3.5.30729; .NET CLR 3.0.30729; .NET CLR 2.0.50727; Media Center PC 6.0)",
    "Mozilla/5.0 (compatible; MSIE 8.0; Windows NT 6.0; Trident/4.0; WOW64; Trident/4.0; SLCC2; .NET CLR 2.0.50727; .NET CLR 3.5.30729; .NET CLR 3.0.30729; .NET CLR 1.0.3705; .NET CLR 1.1.4322)",
    "Mozilla/4.0 (compatible; MSIE 7.0b; Windows NT 5.2; .NET CLR 1.1.4322; .NET CLR 2.0.50727; InfoPath.2; .NET CLR 3.0.04506.30)",
    "Mozilla/5.0 (Windows; U; Windows NT 5.1; zh-CN) AppleWebKit/523.15 (KHTML, like Gecko, Safari/419.3) Arora/0.3 (Change: 287 c9dfb30)",
    "Mozilla/5.0 (X11; U; Linux; en-US) AppleWebKit/527+ (KHTML, like Gecko, Safari/419.3) Arora/0.6",
    "Mozilla/5.0 (Windows; U; Windows NT 5.1; en-US; rv:1.8.1.2pre) Gecko/20070215 K-Ninja/2.1.1",
    "Mozilla/5.0 (Windows; U; Windows NT 5.1; zh-CN; rv:1.9) Gecko/20080705 Firefox/3.0 Kapiko/3.0",
    "Mozilla/5.0 (X11; Linux i686; U;) Gecko/20070322 Kazehakase/0.4.5",
    "Mozilla/5.0 (X11; U; Linux i686; en-US; rv:1.9.0.8) Gecko Fedora/1.9.0.8-1.fc10 Kazehakase/0.5.6",
    "Mozilla/5.0 (Windows NT 6.1; WOW64) AppleWebKit/535.11 (KHTML, like Gecko) Chrome/17.0.963.56 Safari/535.11",
    "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_7_3) AppleWebKit/535.20 (KHTML, like Gecko) Chrome/19.0.1036.7 Safari/535.20",
    "Mozilla/5.0 (Windows NT 6.1; Win64; x64; rv:56.0) Gecko/20100101 Firefox/56.0",
    "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/76.0.3809.100 Safari/537.36",
]
options = webdriver.ChromeOptions()
options.add_experimental_option('excludeSwitches', ['enable-automation'])
options.add_argument('user-agent="%s"' % choice(user_agent))
options.add_argument("--proxy-server=http://202.112.51.45:3128")
def screenShotImg(url):
    """
    截取图片
    :param url:网页url
    :return:
    """
    browser = webdriver.Chrome(options=options)
    try:
        # 调用implicitly_wait()方法实现隐式等待
        # browser.get('http://m.maoyan.com/ajax/detailmovie?movieId=1238837')
        browser.get(url)
        browser.delete_all_cookies()
        for c in cookies:
            cookie = dict(c, **{
                 'domain': "m.meituan.com",
                 'expirationDate': "2147483647",
                 'path': "/",
                 'httpOnly': False,
                 'HostOnly': False,
                 'Secure': False,
            })
            browser.add_cookie(cookie)
        browser.get(url)
        browser.implicitly_wait(60)
        img = browser.find_element_by_id('yodaImgCode')
        screenshot = img.screenshot_as_png
        Image.open(BytesIO(screenshot))
        browser.get_screenshot_as_file('test.png')
        location = img.location
        size = img.size
        # print("浏览器size:", browser.get_window_size())
        # print("全图size:", screenshot.size)
        rangle = (int(location['x'] - 8), int(location['y'] - 8),
                  int(location['x'] + size['width'] + 8),
                  int(location['y'] + size['height'] + 5))
        i = Image.open('test.png')
        frame4 = i.crop(rangle)
        frame4.save('min.png')
        # print(browser.current_url)
        # print(browser.get_cookies())
        # 调用Api识别验证码
        res = get_res('min.png')
        res = json.loads(json.dumps(res))['pic_str']
        print("验证码识别结果：", res)
        # 填充进输入框并触发点击事件
        # javaScript = "document.getElementById('yodaImgCodeInput').value='"+res+"';document.getElementById('yodaImgCodeSure').disabled=false;document.getElementById('yodaImgCodeSure').click();"
        # browser.execute_script(javaScript)
        verify_code = browser.find_element_by_id('yodaImgCodeInput')
        verify_code.send_keys(res)
        sub_btn = browser.find_element_by_id('yodaImgCodeSure')
        sub_btn.click()
        time.sleep(10)
    except Exception as e:
        print(e)
    finally:
        browser.close()



if __name__ == '__main__':
    reqCode = '9686473a7d954c9c83115d919e4e9ef8'
    url = 'https://verify.meituan.com/v2/web/general_page?action=spiderindefence&requestCode=' + reqCode +'&platform=1000&adaptor=auto&succCallbackUrl=https://optimus-mtsi.meituan.com/optimus/verifyResult?originUrl=http%3A%2F%2Fm.maoyan.com%2Fajax%2Fdetailmovie%3FmovieId%3D1238837'
    res = screenShotImg(url)
    # print(res)
    # print("验证码识别结果：", json.loads(json.dumps(res))['pic_str'])
    # 识别成功后调用api提交结果
    # https://verify.meituan.com/v2/ext_api/spiderindefence/verify
    # post
    # id:1
    # request_code:bc9f008845124311bb7df08adeba0e91   _
    # token
    # captchacode:验证码


