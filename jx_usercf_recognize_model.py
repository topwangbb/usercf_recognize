import warnings
import pandas as pd
global net, logger_info
warnings.filterwarnings('ignore')
import numpy as np


def detect_tran(x):
    if str(x).isdigit():
        a = int(x)
        if a < 0:
            return 0
        else:
            return a
    else:
        return 0

def min_max_tran(x):
    return (x - min(x)) /(max(x)- min(x))

def sort_top(x):
    usre_id, score = [], []
    for i in x:
        usre_id.append(str(i[0]))
        score.append(round(i[1], 4))

    return usre_id, score

def Usercf_recognize(df_last, recognize_app_subtype_id, top_n, target_user):
    # 筛选用目标业务服务的客户
    target_all = df_last[df_last['app_subtype_id'] == recognize_app_subtype_id]['msisdn'].unique()

    # 获取没用过目标业务业务的用户
    letent_id = df_last['msisdn'].unique()
    mask = np.isin(letent_id, target_all,invert=True)
    letent_id = letent_id[mask]

    # 删除目标业务的使用记录
    df_last = df_last[df_last['app_subtype_id'] != recognize_app_subtype_id]

    # 极大极小值处理 3σ
    df_last.loc[df_last['ul_data'] > (df_last['ul_data'].mean()+3*df_last['ul_data'].std()),'ul_data'] = df_last['ul_data'].mean()+3*df_last['ul_data'].std()
    df_last.loc[df_last['ul_data'] < (df_last['ul_data'].mean()-3*df_last['ul_data'].std()),'ul_data'] = df_last['ul_data'].mean()-3*df_last['ul_data'].std()
    df_last.loc[df_last['dl_data'] > (df_last['dl_data'].mean()+3*df_last['dl_data'].std()),'dl_data'] = df_last['dl_data'].mean()+3*df_last['dl_data'].std()
    df_last.loc[df_last['dl_data'] < (df_last['dl_data'].mean()-3*df_last['dl_data'].std()),'dl_data'] = df_last['dl_data'].mean()-3*df_last['dl_data'].std()
    df_last.loc[df_last['http_req_nbr'] > (df_last['http_req_nbr'].mean()+3*df_last['http_req_nbr'].std()), 'http_req_nbr'] = df_last['http_req_nbr'].mean()+3*df_last['http_req_nbr'].std()

    # 根据用户id和业务类型聚类求和
    LFM_df = df_last.groupby(['msisdn', 'app_subtype_id'])[['ul_data', 'dl_data', 'http_req_nbr', 'cnt']].sum()

    # 归一化，让模型训练的更快
    for column in ['ul_data', 'dl_data', 'http_req_nbr', 'cnt']:
        LFM_df[column] = min_max_tran(LFM_df[column])

    # 评分矩阵
    f_pivoted = pd.concat([LFM_df.reset_index().pivot(index='msisdn', columns='app_subtype_id', values='ul_data'),
                          LFM_df.reset_index().pivot(index='msisdn', columns='app_subtype_id', values='dl_data'),
                          LFM_df.reset_index().pivot(index='msisdn', columns='app_subtype_id', values='http_req_nbr'),
                          LFM_df.reset_index().pivot(index='msisdn', columns='app_subtype_id', values='cnt')], axis=1)
    # 行为空缺值填补为0
    f_pivoted = f_pivoted.fillna(0)

    score_dict = []
    # 目标的用户画像
    targe_presonas = f_pivoted[f_pivoted.index.isin(target_user)].mean()
    for index2 in letent_id:
        score = np.dot(targe_presonas,f_pivoted.loc[index2])/(np.linalg.norm(targe_presonas)*np.linalg.norm(f_pivoted.loc[index2])) *f_pivoted.loc[index2].sum()
        score_dict.append((index2, score))
    result_df = sorted(score_dict, key = lambda x: x[1], reverse=True)

    # 提取出用户排序
    user, score = sort_top(result_df[:top_n])


    return user, score