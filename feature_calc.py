#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Jun 12 14:19:40 2019

@author: wuzhiqiang
提取所需特征并存储供分析用
"""

print(__doc__)

import sys
import os

app_dir = "./sample_cell"

def init_data_para():
    para_dict = {}
    para_dict['raw_data_dir'] = {'debug': os.path.normpath('./data/processed_data/raw'),
                                 'run': os.path.normpath('/raid/data/raw')}
    para_dict['processed_data_dir'] = {'debug': os.path.normpath(app_dir + '/data'),
                                        'run': os.path.normpath('/raid/data/processed_data/complete')}
    para_dict['scale_data_dir'] = {'debug': os.path.normpath(app_dir + '/data'),
                                    'run': os.path.normpath('/raid/data/processed_data/scale')}
    para_dict['config'] = {'debug': {'s': '192.168.1.105', 'u': 'data', 'p': 'For2019&tomorrow', 'db': 'test_bat', 'port': 3306},
                             'run': {'s': 'localhost', 'u': 'data', 'p': 'For2019&tomorrow', 'db': 'test_bat', 'port': 3306}
                             #'run': {'s': 'localhost', 'u': 'root', 'p': 'wzqsql', 'db': 'cell_lg36', 'port': 3306}
                             }
    para_dict['data_limit'] = {'debug': 100000,
                                 'run': None}
    
    para_dict['score_key'] = 'c'
    para_dict['cell_key'] = 'cell_no'
    para_dict['states'] = ['charge', 'discharge']
    para_dict['log_pro'] = 'processed'
    para_dict['log_info'] = 'pro_info'
    
    para_dict['start_kwd'] = 'start_tick'
    para_dict['end_kwd'] = 'end_tick'
    
    para_dict['run_mode'] = 'debug'
    return para_dict

def main(argv):
    print('starting....')
    sys.path.append(app_dir)
    import rw_bat_data as rwd
    #import preprocess_data as ppd
    import func as fc
    import cell_v_drop as cvd
    
    para_dict =  init_data_para()
    para_dict = fc.deal_argv(argv, para_dict)
    mode = para_dict['run_mode']
    
    print('starting processing the data...')
    bat_list = rwd.get_bat_list(para_dict, mode)
    if bat_list is not None:
        for bat_name in bat_list:
            
            pro_info = rwd.get_pro_info(para_dict, mode, bat_name)
            rwd.save_pro_info(pro_info, para_dict['log_info']+'_'+bat_name, para_dict, mode)
            
            #获得cell_v_drop
            pro_info = rwd.read_bat_data(para_dict, mode, para_dict['log_info']+'_'+bat_name)
            feature = cvd.get_feature(para_dict, mode, bat_name, pro_info)
            rwd.save_pro_info(feature, 'cell_v_drop_'+bat_name, para_dict, mode)
    else:
         print('there is no bat!')
         
if __name__ == '__main__':
    main(sys.argv)