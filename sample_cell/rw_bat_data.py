#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Jun  3 14:18:32 2019

@author: wuzhiqiang

"""
import lib_path
import io_operation as ioo
import processbar as pbar
import g_function as gf

def get_bat_list(para_dict, mode):
    config = para_dict['config'][mode]
    bat_list = ioo.input_table_name(config)
    print(bat_list)
    return bat_list

def read_bat_data(para_dict, mode, bat_name, **kwg):
    config = para_dict['config'][mode]
    kwds = {}
    if 'limit' in kwg:
        kwds['limit'] = kwg['limit']
    if 'start_time' in kwg and 'end_time' in kwg:
        kwds['start_time'] = kwg['start_time']
        kwds['end_time'] = kwg['end_time']
    raw_data = ioo.read_sql_data(config, bat_name, **kwds)
    return raw_data

def save_bat_data(data, filename, para_dict, mode, chunksize=None):
    ioo.save_data_csv(data, filename, para_dict['processed_data_dir'][mode])
    
def save_pro_info(data, table_name, para_dict, mode, if_exists='replace', chunksize=None):
    config = para_dict['config'][mode]
    ioo.save_data_sql(data, config, table_name, chunksize=chunksize)
    
def get_pro_info(para_dict, mode, table_name):
    total = 10
    bar = pbar.Processbar(total)
    showbar = pbar.showbar(bar)
    df = gf.find_sequence(table_name, showbar, total=total, charge_time=8, discharge_time=8, **para_dict)
    bar.finish()
    return df

def get_bat_1_data(para_dict, mode, table_name, condition, str_value):
    """
    获得指定条件的一条电池数据
    """
    config = para_dict['config'][mode]
    df = ioo.match_sql_data(config, table_name, condition, str_value)
    return df.iloc[[0]]

def transfer_data(cnt, data, keywords):
    df = gf.transfer_data(cnt, data, keywords)
    return df
    
def save_train_xlsx(data, filename, output_dir):
    train_feature_gp = data.groupby('section') #按数据区间划分
    feature_dict = {}
    for k in train_feature_gp.groups.keys():
        print('----------------%s----------------'%k)
        feature_dict[k] = train_feature_gp.get_group(k)
    ioo.save_dict_xlsx(feature_dict, filename, output_dir)