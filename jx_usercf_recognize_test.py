# -*- coding: utf-8 -*-
# @Time : 2023/6/21 13:17
# @Author : Wangbb
# @FileName: jx_usercf_recognize_test.py


import requests
import time
import json


payload = {"order_uuid": "123456",
        "recognize_app_subtype_id":3,
               "top_n": 10
}

# 时序预测的示例URL地址
url = 'http://localhost:8888/jx_usercf_recognize'


start = time.time()

# 获取接口
response = requests.post(url,data=payload,files={'file': ('data.csv', open("../app/test_data.csv", 'rb'))})
print(response)
print(time.time()-start)

response.encoding = 'utf-8'
# 吧csv和返回信息一起在content输出为json
result = response.json()
print(result)
print(json.dumps(result, ensure_ascii=False, indent=4))
