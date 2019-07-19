#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Jun  3 13:14:29 2019

@author: wuzhiqiang
"""
import re
import lib_path

def deal_argv(argv, para_dict):
    """
    第一位代表运行模式，debug/run
    """
    para_len = len(argv)
    if para_len  >= 2:
        import g_function as gf
        para_dict['run_mode'] = gf.deal_1_argv(argv)    
        if para_len > 2:
            print("dealing others parameters...")
            para_dict = deal_other_argv(argv, para_dict)  
    return para_dict

def deal_other_argv(argv, para_dict):
    """
    1:bat_mode; 2:bat_type; 3:bat-structure; 4:bat-year; 5:score_key
    """
    i = 0
    mission = ['all', 'pro_info', 'cell_v_drop', 'cell_stdv', 'soh_feature']
    for ar in argv:
        if i == 2: #
            regx = r'\-[a-zA-Z\_]{3,20}'
            if re.match(regx, ar):
                if ar[1:] in mission:
                    para_dict['mission'].append(ar[1:])
                    print('The 2nd input parameter is accepted.')
            else:
                print("The 2nd input parameter '%s' is not accepted."%ar)
        if i == 3: #
            regx = r'\-[a-zA-Z\_]{3,20}'
            if re.match(regx, ar):
                if ar[1:] in mission:
                    para_dict['mission'].append(ar[1:])
                    print('The 3rd input parameter is accepted.')
            else:
                print("The 3rd input parameter '%s' is not accepted."%ar)
        if i == 4: #
            regx = r'\-[a-zA-Z\_]{3,20}'
            if re.match(regx, ar):
                if ar[1:] in mission:
                    para_dict['mission'].append(ar[1:])
                    print('The 4th input parameter is accepted.')
            else:
                print("The 4th input parameter '%s' is not accepted."%ar)
        i += 1
    print(para_dict['mission'])
    return para_dict

import g_function as gf
def get_filename_regx(log, **kwds):
    return gf.get_filename_regx(log, **kwds)

def save_workstate_data(regx, mask_filename, raw_data_dir, data_dir):
    result, temp = gf.get_regx_data(regx, raw_data_dir)
    if result:
        import io_operation as ioo
        data = temp[temp['current_mean'] == 0].reset_index(drop=True) #静置数据
        filename = 'rest_' + mask_filename
        ioo.save_data_csv(data, filename, data_dir)
        data = temp[temp['current_mean'] > 0].reset_index(drop=True) #充电数据
        filename = 'charge_' + mask_filename
        ioo.save_data_csv(data, filename, data_dir)
        data = temp[temp['current_mean'] < 0].reset_index(drop=True) #放电数据
        filename = 'discharge_' + mask_filename
        ioo.save_data_csv(data, filename, data_dir)
        print('the data has been save within each workstate.')
    return result

import feature_function as ff
def get_bat_config(config, cell_info, fuzzy):
    bat_config = ff.get_cell_rate_para(config, cell_info, fuzzy)
    return bat_config