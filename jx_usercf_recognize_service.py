from datetime import datetime
from flask import request, Flask
import traceback
import logging
import html
from utils import logger
from utils.ErrorCode import *
from utils.get_config import *
from model.jx_usercf_recognize_model import detect_tran ,Usercf_recognize
global net, logger_info
import warnings
import pandas as pd
warnings.filterwarnings('ignore')
app = Flask(__name__)
import json


#在视图函数执行后执行
@app.after_request
def apply_caching(response):
    response.headers["X-Frame-Options"] = "DENY"
    response.headers["X-Content-Type-Options"] = "nosniff"
    response.headers['Content-Type'] = 'application/json'
    return response

@app.route('/jx_usercf_recognize', methods=['POST'])
def tower_recognize():
    result = []
    err_msg = None
    code = 0

    if not request.form:
        logger_info.error("Missing input paramas! ")
        raise MissingInputField()

    # 检测参数是否都在form中，文件是否在file中
    if "order_uuid" not in request.form \
            or "recognize_app_subtype_id" not in request.form \
            or "top_n" not in request.form:
        logger_info.error("Missing input paramas! ")
        raise MissingInputField()

    elif request.files:
        if not request.files['file']:
            logger_info.error("Missing input data! ")
            raise MissingInputField()

    # 检查参数是否格式正确，并进行转换
    try:
        order_uuid = html.escape(str(request.form["order_uuid"]), quote=True)
        recognize_app_subtype_id = int(request.form["recognize_app_subtype_id"])
        top_n = int(request.form["top_n"])
    except:
        logger_info.error("Wrong params format! ")
        raise WrongInputType()

    # 对file中的csv转为dataframe,并进行异常值转换
    try:
        file = request.files.get('file')
        df = pd.read_csv(file, encoding='UTF-8')
        df['dl_data'] = df['dl_data'].apply(lambda x: detect_tran(x))
        df['ul_data'] = df['ul_data'].apply(lambda x: detect_tran(x))
        df = df.drop_duplicates()
    except:
        logger_info.error("Wrong csv format! ")
        raise WrongInputType()

    # 去除只有一条使用数据的目标业务用户,如果少于10个用户则报错
    target_user = df[df['app_subtype_id'] == recognize_app_subtype_id]['msisdn'].unique()
    target_count = df[df['msisdn'].isin(target_user)].groupby(['msisdn'])['app_subtype_id'].unique().apply(len)
    target_user = target_count[target_count > 1].index

    if len(target_user) < 10:
        logger_info.error("Loss compelte data! ")
        raise WrongDataloss()

    # 能力推理部分
    try:
        starttime = datetime.now()
        # 调用模型
        user, score = Usercf_recognize(df, recognize_app_subtype_id, top_n, target_user)
        endtime = datetime.now()
        code = 200
        logger_info.info("spend %f seconds this time with result %s." %
                         ((endtime - starttime).total_seconds(), result))

    except Exception as e:
        # 把报错的完整内容保存到日志文件中
        traceback.print_exc()
        code = -1
        err_msg = str(e)
        #输出ERROR:err_msg
        logger_info.error(err_msg)

    # 无论try语句是否抛出异常，finally中的语句一定会被执行
    finally:
        try:
            if err_msg is None:
                result = {
                    "code": code,
                    "order_uuid": order_uuid,
                    "recognize_app_subtype_id": 3,
                    "msg": "Successful",
                    "user_top": user,
                    "user_score": score
                }
            else:
                result = {
                    "code": code,
                    "order_uuid": order_uuid,
                    "msg": err_msg
                }
        except:
            pass

    return json.dumps(result)


if __name__ == "__main__":
    # 读取能力基本配置
    CONFIG = get_config('configuration')


    # 输出json日志形式
    logger_info = logger.JsonLogger(datetime.now().strftime('_%Y%m%d%H%M%S')).getLogger()
    # 设置日志级别
    logger_info.setLevel(logging.DEBUG)

    # 启动flask服务
    app.run(host=CONFIG["host"], port=int(CONFIG["port"]))