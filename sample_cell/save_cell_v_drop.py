#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Jun 16 18:46:03 2019

@author: wuzhiqiang
"""

import pandas as pd
import rw_bat_data as rwd

para_dict = {}
para_dict['config'] = {'debug': {'s': '192.168.1.105', 'u': 'data', 'p': 'For2019&tomorrow', 'db': 'test_bat', 'port': 3306},
                             'run': {'s': 'localhost', 'u': 'data', 'p': 'For2019&tomorrow', 'db': 'test_bat', 'port': 3306}
                             }
para_dict['processed_data_dir'] = {'debug': './data'}
mode = 'debug'
table_list = rwd.get_bat_list(para_dict, mode)
if table_list is not None:
    v_drop = []
    for table_name in table_list:
        v = rwd.read_bat_data(para_dict, mode, table_name)
        v['bat_id'] = table_name
        v = v.drop(columns='index')
        v_drop.append(v)
    df = pd.concat(tuple(v_drop))
    rwd.save_bat_data(df, 'cell_v_drop', para_dict, mode)