from flask import Flask
from flask import request
import requests
import json

app = Flask(__name__)

client_id='2d055849ef325d0da767'
client_secret='e6f42f0be63877b065c8523c1e6d48bdad00a9ba'


@app.route('/oauth/callback', methods=['GET', 'POST'])
def oauth_callback():
   oauth_code = request.args['code']
   # 请求access_token
   url = 'https://github.com/login/oauth/access_token?client_id='+client_id+'&client_secret='+client_secret+'&code='+oauth_code
   headers = {
       'accept': 'application/json',
   }
   response = requests.post(url, headers=headers)
   response_content_json = json.loads(str(response.content,'utf-8'))
   # 获取access_token
   access_token=response_content_json['access_token']
   # 请求用户数据（携带access_token）
   url = 'https://api.github.com/user'
   headers = {
       'accept': 'application/json',
       'Authorization':'token '+access_token
   }
   response = requests.get(url, headers=headers)
   return str(response.content,'utf-8')



if __name__ == '__main__':
    app.run(host='0.0.0.0',port=8080,debug=True)

