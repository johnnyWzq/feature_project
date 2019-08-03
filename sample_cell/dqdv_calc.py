#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Jul 19 13:24:32 2019

@author: wuzhiqiang
"""
import pandas as pd
import numpy as np
import re
#import soh_feature as sf
import soh_feature_1 as sf1
import rw_bat_data as rwd
import datetime

def normalize_feature(data, V_RATE, v_cols):
    for col in data.columns:
        for j in v_cols:
            if re.match(j, col):
                data[col+'_nml'] = data[col] / V_RATE
    return data

def calc_dqdv(df, bias, C_RATE, sample=None, parallel=1):
    """
    #计算dq/dv，由于没有dq，使用i代替，但需要考虑采样频率换算成1s
    #parallel指的是系统中pack的并联数
    """
    if len(df) <= 1:
        return 88888888
    df = df.drop_duplicates('stime')
    if sample is None:
        start_time = datetime.datetime.strptime(df['stime'].iloc[0][:19], "%Y-%m-%d %H:%M:%S")
        end_time = datetime.datetime.strptime(df['stime'].iloc[1][:19], "%Y-%m-%d %H:%M:%S")
        sample = (end_time - start_time).seconds
    dqdv = (df['current'].iloc[bias:].sum() / parallel * sample) / (df['voltage'].iloc[-1] - df['voltage'].iloc[0])
    #dqdv = len(df) / (df['voltage'].iloc[-1] - df['voltage'].iloc[0])
    dqdv /= C_RATE
    regular = 0
    #regular = regular_dqdv(abs(df['current'].mean())/C_RATE)
    dqdv = dqdv * (1 + regular)
    if dqdv == np.inf or dqdv == -np.inf or dqdv <= 0: #电压变化较快，一条数据就超过设定值
        dqdv = 88888888
    return dqdv

def find_ic(df, C_RATE, cnt, sample_time, parallel, series, start_value, direction):
    delta_v = 0.01 * series
    clip_data_list, pos_seq = sf1.slip_data_by_volt(df, delta_v, method=1, start_value= start_value, direction=direction)
    total_data = pd.DataFrame()
    dqdv_list = []
    j = 0
    for clip_data in clip_data_list:
        dqdv = calc_dqdv(clip_data, 0, C_RATE, sample=sample_time, parallel=parallel)
        if dqdv != 88888888:
            dqdv_list.append(dqdv)
            total_data = total_data.append(rwd.transfer_data(j, clip_data[['voltage', 'stime', 'c']], keywords='stime')) #获得每一个电压数据片对电压的统计值
            j += 1 #只有当dqdv有效，j才增加
    del clip_data_list
    dqdv_list = sf1.outlier_err_dqdv(dqdv_list)#对异常点进行替换
    if dqdv_list is None:
        return None
    total_data['dqdv'] = dqdv_list
    total_data['voltage_mean'] = total_data['voltage_mean'].round(2)
    return total_data[['dqdv', 'voltage_mean']]

def get_dqdv_data(para_dict, mode, bat_name, pro_info, keywords='voltage'):
    pro_info = pro_info[pro_info['state'] != 0]
    if len(pro_info) < 1:
        return None
    sample_time = pro_info['sample_time'].iloc[0]
    C_RATE = para_dict['bat_config']['C_RATE']
    V_RATE = para_dict['bat_config']['V_RATE']
    T_REFER = para_dict['bat_config']['T_REFER']
    bat_type = para_dict['bat_config']['bat_type']
    parallel = para_dict['bat_config']['parallel']
    series = para_dict['bat_config']['series']
    border_dict = {'min': [0, 2.5, 2.5], 'max': [0, 4.3, 4.3]}
    print('starting calculating the features of battery for soh...')
    dqdv_data = pd.DataFrame()
    for i in range(1400, 1404):#len(pro_info)):
        print('-----------------round %d-------------------'%i)
        state = pro_info['state'].iloc[i]
        df = sf1.get_1_pro_data(para_dict, mode, bat_name, pro_info, i)
        df = df.reset_index(drop=True)
        df = sf1.calc_other_vectors(df, state)
        start_value = sf1.get_start_value(border_dict, state)
        feature_df = find_ic(df, C_RATE, i, sample_time, parallel, series, start_value, state)
        if feature_df is None:
            continue
        #feature_df = normalize_feature(feature_df, V_RATE, ['voltage'])
        feature_df = feature_df.set_index('voltage_mean', drop=True)
        feature_df = feature_df.rename(columns={'dqdv': str(state)+'_'+str(pro_info['process_no'].iloc[i])})
        feature_df = feature_df.sort_index()
        del df
        dqdv_data = dqdv_data.merge(feature_df, left_index=True, right_index=True, how='outer')
    #dqdv_data = pd.concat(tuple(dqdv_data))
    #dqdv_data = normalize_feature(dqdv_data, V_RATE, ['voltage'])
    #dqdv_data = dqdv_data.reset_index(drop=True)
    rwd.save_bat_data(dqdv_data, 'cell_dqdv_'+bat_name, para_dict, mode)
    return dqdv_data

