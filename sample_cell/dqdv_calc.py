#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Jul 19 13:24:32 2019

@author: wuzhiqiang
"""
import pandas as pd
import re
#import soh_feature as sf
import soh_feature as sf1
import rw_bat_data as rwd

def normalize_feature(data, V_RATE, v_cols):
    for col in data.columns:
        for j in v_cols:
            if re.match(j, col):
                data[col+'_nml'] = data[col] / V_RATE
    return data

def find_ic(df, C_RATE, cnt, sample_time, parallel, series, start_value, direction)):
    delta_v = 0.01 * series
    clip_data_list, pos_seq = sf1.clip_data_list, pos_seq = slip_data_by_volt(df, delta_v, method=1, start_value= start_value, direction=direction)
    total_data = pd.DataFrame()
    dqdv_list = []
    j = 0
    for clip_data in clip_data_list:
        dqdv = sf1.calc_dqdv(clip_data, 0, C_RATE, sample=sample_time, parallel=parallel)
        if dqdv != 88888888:
            dqdv_list.append(dqdv)
            total_data = total_data.append(rwd.transfer_data(j, clip_data[['voltage', 'stime', 'c']], keywords='stime')) #获得每一个电压数据片对电压的统计值
            j += 1 #只有当dqdv有效，j才增加
    del clip_data_list
    dqdv_list = sf1.outlier_err_dqdv(dqdv_list)#对异常点进行替换
    total_data['dqdv'] = dqdv_list
    return total_data[['dqdv', 'voltage_mean', 'voltage_mean_nml']]

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
    border_dict = dict('min'=[2.5, 2.5], 'max'=[4.2, 4.2])
    print('starting calculating the features of battery for soh...')
    dqdv_data = []
    for i in range(0, len(pro_info), 100):
        print('-----------------round %d-------------------'%i)
        state = pro_info['state'].iloc[i]
        df = sf1.get_1_pro_data(para_dict, mode, bat_name, pro_info, i)
        df = df.reset_index(drop=True)
        df = sf1.calc_other_vectors(df, state)
        start_value = get_start_value(border_dict, state)
        feature_df = find_ic(df, C_RATE, i, sample_time, parallel, series, start_value, state)
        if feature_df is None:
            continue
        feature_df = normalize_feature(feature_df, V_RATE, ['voltage'])
        feature_df = feature_df.T
        del df
        feature_df['state'] = state
        feature_df['process_no'] = pro_info['process_no'].iloc[i]
    dqdv_data.append(feature_df)
    if dqdv_data == []:
        return None
    dqdv_data = pd.concat(tuple(dqdv_data))
    dqdv_data = normalize_feature(dqdv_data, V_RATE, ['voltage'])
    dqdv_data = dqdv_data.reset_index(drop=True)
    rwd.save_bat_data(dqdv_data, 'cell_dqdv_'+bat_name, para_dict, mode)
    return dqdv_data

