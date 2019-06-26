#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Apr  3 15:45:55 2019

@author: wuzhiqiang
"""

import os
import pandas as pd
import time
from dateutil import parser

import lib_path
import g_function as gf
     
def clean_data(data):
    
    TEMP_DEFAULT = 25
    #如果temperature是空，填默认值25
    data.loc[:, 'temperature'] = data['temperature'].fillna(method='ffill').fillna(method='bfill').fillna(TEMP_DEFAULT)
    #data['temperature'] = data['temperature'].where(~data['temperature'].isna(), TEMP_DEFAULT)
    #filt samples on rows, if a row has too few none-nan value, drop it
    DROPNA_THRESH = 5
    data = data.dropna(thresh=DROPNA_THRESH)
    data = data[['timestamp', 'stime', 'cycle_no', 'step_no', 'current', 'voltage',
                 'charge_c', 'discharge_c', 'temperature']]
    return data

def calc_other_vectors(df0):
    """
    #计算dq/dv值
    #由于没有dq值，因此使用i代替
    #计算i/dv
    """
    df = df0.copy()
    df.loc[:, 'dqdv'] = df['current'] / df['voltage'].diff()
    if (df['current'].mean(skipna=True)) > 0: #充电
        df.loc[:, 'c'] = df['charge_c']
    elif (df['current'].mean(skipna=True)) < 0: #放电
        df.loc[:, 'c'] = df['discharge_c']
    else:
        df.loc[:, 'c'] = 0
    return df
   
def add_ocv_c(data):
    """
    #静置时c等于上一次充/放电容量与后一次充/放电容量的均值
    #第一次及最后一次不计算
    """
    '''
    if len(data) > 2:
        data = data.drop(index=0, axis=1).drop(index=len(data)-1, axis=1)
    '''
    for i in data[data['c'] == 0].index:
        if (i == 0) or (i == len(data)-1):
            continue
        data.loc[i, 'c'] = 0.5 * (data.loc[i-1, 'c'] + data.loc[i+1, 'c'])
    return data

def slip_data(df):
    """
    #按step_no将数据进行切片，再按每个过程切片
    """
    PROCESS_GAP = 10 #10points,10sec
    PROCESSING_GAP = 10
    data0 = pd.DataFrame()
    start = time.time()
    for value in set(df['step_no'].tolist()):#到所有都处理完了最后再做一次排序
        idx = df[df['step_no'] == value].index
        data = pd.DataFrame()
        j_last = 0
        cnt = 0
        for j in range(1, len(idx) + 1):
            if j >= len(idx) or idx[j] - idx[j - 1] > PROCESS_GAP:    
                cur_df = df.loc[idx[j_last]:idx[j-1]] #idx[x]代表df的index值,所以用loc
                print('clip %d : j: %d -> %d, the length of cur_df: %d.'
                      %(cnt, idx[j_last], idx[j-1], len(cur_df)))
                j_last = j
                if len(cur_df) < PROCESSING_GAP:
                    continue
                cur_df = calc_other_vectors(cur_df)
                
                data = data.append(gf.transfer_data(cnt, cur_df))
                cnt += 1
        data0 = data0.append(data)
    end = time.time()
    print('Done, it took %d seconds to sliping the data.'%(end-start))
    return data0
                
def find_dcr_process(data, key):
    """
    找到所有充、放、静置过程并计算每个过程的其他参数
    再将每个过程压缩成一行数据
    """
    temp = data[[key]]
    j_last = 0
    cnt = 0  
    start = time.time()
    df = pd.DataFrame()
    for j in range(1, len(temp) + 1):
        if j >= len(temp) or temp[key].iloc[j] < temp[key].iloc[j - 1]:#直到下一个实验的循环计数开始
            #有些实验cycle_no是分段重新计数，需要找到并拼接
            if j_last == 0:
                bias = 0
            else:
                bias = data[key].iloc[j_last - 1]
            cur_df = data.iloc[j_last:j]
            cur_df.loc[:, key] = cur_df[key] + bias
            print('clip %d : j: %d -> %d, the length of current_df: %d.'%(cnt, j_last, j, len(cur_df)))
            j_last = j
            cnt += 1
            df = df.append(slip_data(cur_df))
    df.loc[:, 'start_tick'] = df['start_tick'].apply(str)
    df.loc[:, 'start_tick'] = df['start_tick'].apply(lambda x: parser.parse(x))
    df = df.sort_values('start_tick')
    end = time.time()
    print('Done, it took %d seconds to finishing the job.'%(end-start))
    return df

def preprocess_data(cell_info, data0=None):
    """
    将预处理后的数据按一次充电过程进行分割合并
    """
    #regular the cycle_no
    data0 = clean_data(data0)

    data = find_dcr_process(data0, 'cycle_no')
    
    data.insert(0, 'cell_no', cell_info)

    data = data.reset_index(drop=True)
    data = add_ocv_c(data)
    return data


def main():
    data_dir = os.path.join(os.path.abspath('.'), 'data')
    cell_no = '29'
    temperature = '35'
    cycle = '0-1000'
    filename = 'LG36-%s-%s_%s'%(temperature, cell_no, cycle)
    """
    data_ori_dir = os.path.normpath('/Volumes/Elements/data/电池数据')#('/Users/admin/Documents/data/电池数据')
    data_ori = rd.read_data(data_ori_dir, cell_no, temperature)
    data = rd.clean_data(data_ori)
    rd.save_data_csv(data, filename, data_dir, 500000)
    """
    #"""
    #"""
    #rd.save_workstate_data(r'processed_LG36-\d+-\d+_\d-\d+.csv', data_dir)
    
if __name__ == '__main__':
    main()