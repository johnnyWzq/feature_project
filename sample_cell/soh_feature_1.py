#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Jul 13 22:58:48 2019

@author: wuzhiqiang
"""

#!/usr/bin/env python3

import pandas as pd
import numpy as np
import re
import rw_bat_data as rwd
import datetime

    
def get_1_pro_data(para_dict, mode, table_name, pro_info, pro_no, condition1='start_time', condition2='end_time', t_keywords='stime'):
    str_value1 = pro_info[condition1].iloc[pro_no] 
    str_value1 = str_value1.strftime("%Y-%m-%d %H:%M:%S")
    str_value2 = pro_info[condition2].iloc[pro_no] 
    str_value2 = str_value2.strftime("%Y-%m-%d %H:%M:%S")
    df = rwd.read_bat_data(para_dict, mode, table_name, start_time=str_value1, end_time=str_value2)
    return df

def slip_data_by_volt(df, delta_v=0.01, keywords='voltage', method=0, start_value= 2.7, direction=1):
    """
    #按delta_v进行切片，series为电芯串联数
    #direction=1为数值增加
    """
    if df is None or len(df) <= 1:
        print('there is not enough data.')
        return None, None
    if method == 0:
        start = 0
        clip_data_list = []
        pos_seq = [start]
        for i in range(1, len(df)):
            if abs((df.iloc[i][keywords] - df.iloc[start][keywords])) >= delta_v or \
                i == (len(df) - 1):
                if i == (len(df) - 1):
                    clip_data_list.append(df.iloc[start:])
                else:
                    clip_data_list.append(df.iloc[start: i])
                    start = i
                    pos_seq.append(start)
    elif method == 1:
        if direction == 1:
            border_list = [(start_value + delta_v * i) for i in range(200)]
            for i, tmp in enumerate(border_list):
                if df.iloc[0][keywords] >= tmp and df.iloc[0][keywords] <= border_list[i + 1]:
                    border = tmp
                    break
        elif direction == 2:
            border_list = [(start_value - delta_v * i) for i in range(200)]
            for i, tmp in enumerate(border_list):
                if df.iloc[0][keywords] <= tmp and df.iloc[0][keywords] >= border_list[i + 1]:
                    border = tmp
                    break
        start = 0
        clip_data_list = []
        pos_seq = [start]
        for i in range(1, len(df)):
            if abs((df.iloc[i][keywords] - border)) >= delta_v or i == (len(df) - 1):
                if i == (len(df) - 1):
                    clip_data_list.append(df.iloc[start:])
                else:
                    clip_data_list.append(df.iloc[start: i])
                    border = df.iloc[i][keywords]
                    start = i
                    pos_seq.append(start)
    return clip_data_list, pos_seq

def normalize_feature(data, V_RATE, C_RATE, T_RATE, v_cols, c_cols, t_cols):
    for col in data.columns:
        for j in v_cols:
            if re.match(j, col):
                data[col] = data[col] / V_RATE
        for j in c_cols:
            if re.match(j, col):
                data[col] = data[col] / C_RATE
        for k in t_cols:
            if re.match(k, col):
                data[col] = data[col] / T_RATE
    return data

def calc_other_vectors(df, state):
    """
    """
    if state == 1: #充电
        df['c'] = df['charge_c']
        #df['e'] = df['charge_e']
    elif state == 2: #放电
        df['c'] = df['discharge_c']
        #df['e'] = df['discharge_e']
    else:
        df['c'] = 0
        #df['e'] = 0
    return df

def regular_dqdv(rate):
    bias = 0
    if rate <= 0.3:
        bias = 0.02
    elif rate > 0.3 and rate <= 0.7:
        bias = 0.01
    elif rate > 0.7 and rate <= 1.2:
        bias = 0
    elif rate > 1.2 and rate <= 1.5:
        bias = -0.01
    elif rate > 1.5 and rate <= 2:
        bias = -0.02
    else:
        bias = -0.03
    return bias
        
def calc_dqdv(df, bias, C_RATE, sample=None, parallel=1):
    """
    #计算dq/dv，由于没有dq，使用i代替，但需要考虑采样频率换算成1s
    #parallel指的是系统中pack的并联数
    """
    if len(df) <= 5:
        return 88888888, 88888888
    df = df.drop_duplicates('stime')
    if sample is None:
        start_time = datetime.datetime.strptime(df['stime'].iloc[0][:19], "%Y-%m-%d %H:%M:%S")
        end_time = datetime.datetime.strptime(df['stime'].iloc[1][:19], "%Y-%m-%d %H:%M:%S")
        sample = (end_time - start_time).seconds
    dqdv = (df['current'].iloc[bias:].sum() / parallel * sample) / (df['voltage'].iloc[-1] - df['voltage'].iloc[0])
    #dqdv = len(df) / (df['voltage'].iloc[-1] - df['voltage'].iloc[0])
    dqdv /= C_RATE
    bias = regular_dqdv(abs(df['current'].mean())/C_RATE)
    dqdv_fix = dqdv * (1 + bias)
    if dqdv == np.inf or dqdv == -np.inf or dqdv <= 0: #电压变化较快，一条数据就超过设定值
        dqdv, dqdv_fix = 88888888, 88888888
    return dqdv, dqdv_fix

def outlier_err_dqdv(dqdv_list, method=2):
    """
    #1种是比较平均值找异常
    #2种是比较相邻，相差太大异常
    """
    if len(dqdv_list) == 0:
        return None
    if method == 1:
        outlier = 3
        dqdv_mean = 0
        for dqdv in dqdv_list:
            dqdv_mean += dqdv
        dqdv_mean /= len(dqdv_list)
        for dqdv in dqdv_list:
            if dqdv >= (dqdv_mean * outlier):
                dqdv_list.remove(dqdv)
    elif method == 2:
        outlier = 8
        dqdv_diff_list = []
        for i in range(1, len(dqdv_list)):
            dqdv_diff_list.append(abs(dqdv_list[i] - dqdv_list[i - 1]))
        if len(dqdv_diff_list) == 0:
            return None
        dqdv_diff_mean = 0
        for dqdv_diff in dqdv_diff_list:
            dqdv_diff_mean += dqdv_diff
        dqdv_diff_mean /= len(dqdv_diff_list)
        for dqdv_diff in dqdv_diff_list:
            if dqdv_diff > (dqdv_diff_mean * outlier): #将这个异常点替换成前一点加上前一个diff
                dqdv_diff_err_index = dqdv_diff_list.index(dqdv_diff)#得到此异常点前一个点位置
                dqdv_list.insert(dqdv_diff_err_index + 1, dqdv_diff_list[dqdv_diff_err_index - 1] + dqdv_list[dqdv_diff_err_index])
                del dqdv_list[dqdv_diff_err_index + 2] #删除
                
    return dqdv_list
    
def get_dqdv_incline(data, col_name):
    data[col_name + '_incline'] = 1
    for i in range(1, len(data)):
        data[col_name + '_incline'].iloc[i] = data[col_name].iloc[i] / data[col_name].iloc[0]
    return data
    
def find_ic_feature(df, C_RATE, cnt, sample_time, parallel, series, start_value, direction):
    """
    """
    delta_v = 0.01 * series
    clip_data_list, pos_seq = slip_data_by_volt(df, delta_v, method=1, start_value= start_value, direction=direction)
    if clip_data_list is None or pos_seq is None:
        return None
    total_data = pd.DataFrame()
    dqdv_list = []
    dqdv_fix_list = []
    j = 0
    for clip_data in clip_data_list:
        dqdv, dqdv_fix = calc_dqdv(clip_data, 0, C_RATE, sample=None, parallel=parallel)
        if dqdv != 88888888:
            dqdv_list.append(dqdv)
            dqdv_fix_list.append(dqdv_fix)
            total_data = total_data.append(rwd.transfer_data(j, clip_data, keywords='stime')) #获得每一个电压数据片对电压的统计值
            j += 1 #只有当dqdv有效，j才增加
    del clip_data_list
    dqdv_list = outlier_err_dqdv(dqdv_list)#对异常点进行替换
    dqdv_fix_list = outlier_err_dqdv(dqdv_fix_list)
    if dqdv_list is None:
        return None
    total_data['dqdv'] = dqdv_list
    total_data = get_dqdv_incline(total_data, 'dqdv')
    total_data['dqdv_fix'] = dqdv_fix_list
    total_data = get_dqdv_incline(total_data, 'dqdv_fix')
    sel_cols = ['start_tick', 'data_num', 'dqdv', 'dqdv_fix', 'dqdv_fix_incline', 'dqdv_incline', 'voltage_mean', 'voltage_std', 'voltage_diff_mean', 'voltage_diff_std', 'temperature_mean', 'c']
    total_data = total_data[sel_cols]
    total_data = total_data.rename(columns={'data_num': 'clip_num'})
    #total_data['start_tick'] = total_data['start_tick'].apply(str)
    total_data = rwd.transfer_data(cnt, total_data, keywords='start_tick')
    return total_data

def find_border(V_RATE, bat_type='NCM'):
    """
    cell_para = [[3.495, 3.620, 3.67, 3.702, 3.738, 3.78, 3.843, 3.928, 4.029, 4.143, 4.2],
                 [2.7, 3.354, 3.455, 3.506, 3.544, 3.583, 3.634, 3.706, 3.799, 3.904, 4.049]]
    soc_section = [0, 10, 20, 30, 40, 50, 60, 70, 80, 90, 100]
    border_df = pd.DataFrame(cell_para, columns=soc_section, index=[1, 2])
    border_df = border_df.T
    return border_df
    """
    border_dict = {}
    if bat_type == 'NCM':
        border_dict['min'] = [0, V_RATE * 1.03, V_RATE * 1.02] #[x, charge, discharge]
        border_dict['max'] = [0, V_RATE * 1.1, V_RATE * 1.09]
    return border_dict

def get_start_value(border_dict, state):
    if state == 1:
        start_value = border_dict['min'][state]
    elif state == 2:
        start_value = border_dict['max'][state]
    return start_value

def get_valid_data(df, state, border_min, border_max):
    """
    """
    train_data = df[df['voltage'] >= border_min[state]]
    train_data = train_data[train_data['voltage'] <= border_max[state]]
    return train_data

def generate_train_data(train_data, state, border_min, border_max, bias=0.02):
    """
    """
    train_data_dict = {'%.3f-%.3f'%(border_min[state], border_max[state]): train_data}
    """
    tmp_min = [i * (1 + bias) for i in border_min]
    train_data_dict['%.3f-%.3f'%(tmp_min[state], border_max[state])] = get_valid_data(train_data, state, tmp_min, border_max)
    tmp_max = [i * (1 - bias) for i in border_max]
    train_data_dict['%.3f-%.3f'%(border_min[state], tmp_max[state])] = get_valid_data(train_data, state, border_min, tmp_max)
    """
    return train_data_dict

def get_feature_soh(para_dict, mode, bat_name, pro_info, keywords='voltage'):
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
    border_dict = find_border(V_RATE, bat_type)
    print('starting calculating the features of battery for soh...')
    train_feature = []
    for i in range(0, len(pro_info), 1):
        print('-----------------round %d-------------------'%i)
        state = pro_info['state'].iloc[i]
        df = get_1_pro_data(para_dict, mode, bat_name, pro_info, i)
        df = df.reset_index(drop=True)
        df = calc_other_vectors(df, state)
        cycle_soh = df['c'].iloc[-1]
        train_data = get_valid_data(df, state, border_dict['min'], border_dict['max'])#生产需要的训练数据
        del df
        start_value = get_start_value(border_dict, state)
        train_data_dict = generate_train_data(train_data, state, border_dict['min'], border_dict['max'], 0.02)
        for key, train_data in train_data_dict.items():
            feature_df = find_ic_feature(train_data, C_RATE, i, sample_time, parallel, series, start_value, state)
            if feature_df is None:
                continue
            feature_df['section'] = key
            feature_df['state'] = state
            feature_df['process_no'] = pro_info['process_no'].iloc[i]
            feature_df['soh'] = cycle_soh / C_RATE
            train_feature.append(feature_df)
        del train_data_dict
    if train_feature == []:
        return None
    train_feature = pd.concat(tuple(train_feature))
    train_feature = normalize_feature(train_feature, V_RATE, C_RATE, T_REFER, ['voltage'], ['c'], ['temperature'])
    train_feature = train_feature.reset_index(drop=True)
    rwd.save_bat_data(train_feature, 't_cell_soh_'+bat_name, para_dict, mode)
    return train_feature

def test():
    data = [1,2,3,4,5,4,5,7,9,11,13,9.5,10,10.5,11,11,10.5,11,13,14,15]
    data = pd.DataFrame(data, columns=['dqdv'])
    border_df = find_border()
    generate_train_data(data, 1, border_df, 40, 70)
if __name__ == '__main__':
    test()