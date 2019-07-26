#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Jun 16 18:46:03 2019

@author: wuzhiqiang
"""
import os
import pandas as pd
import rw_bat_data as rwd

para_dict = {}
para_dict['config'] = {'debug': {'s': '192.168.1.105', 'u': 'data', 'p': 'For2019&tomorrow', 'db': 'test_bat', 'port': 3306},
                             'run': {'s': 'localhost', 'u': 'data', 'p': 'For2019&tomorrow', 'db': 'test_bat', 'port': 3306}
                             }
para_dict['processed_data_dir'] = {'debug': './data'}
mode = 'debug'
table_list = rwd.get_bat_list(para_dict, mode)
if 'cell_v_drop' in table_list[0]:
    file_name = 'cell_v_drop'
    cut = 12
elif 'cell_stdv' in table_list[0]:
    file_name = 'cell_stdv'
    cut = 10
elif 'cell_soh' in table_list[0]:
    file_name = 'cell_soh'
    cut = 9
if table_list is not None:
    data = []
    data_dir = para_dict['processed_data_dir'][mode]
    if not os.path.exists(data_dir):
        os.mkdir(data_dir)
    writer = pd.ExcelWriter(os.path.join(data_dir, file_name+'.xlsx'))
    for table_name in table_list:
        v = rwd.read_bat_data(para_dict, mode, table_name)
        v['bat_id'] = table_name[cut:]
        v = v.drop(columns='index')
        v.to_excel(writer, table_name[cut:])
        rwd.save_bat_data(v, table_name, para_dict, mode)
        data.append(v)
    df = pd.concat(tuple(data))
    rwd.save_bat_data(df, file_name, para_dict, mode)