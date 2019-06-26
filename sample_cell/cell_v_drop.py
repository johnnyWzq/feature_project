#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Jun 12 14:19:40 2019

@author: wuzhiqiang
提取极化压差特征，供内短路分析用
"""

print(__doc__)
import datetime
import pandas as pd
import rw_bat_data as rwd

def find_r_d_sequence(state_list):
    """
    找到state_list字段中为（0-2-0）或（0-2-。。。-2-0）的序列
    找到state_list字段中为（0-1-0）或（0-1-。。。-1-0）的序列
    """
    x_len = len(state_list)-1
    if x_len <=2:
        print('there is not enough data to sequenced.')
        return []
    x2_position = []  #0-2-0
    y2_position = []
    x1_position = []  #0-1-0
    y1_position = []
    for i in range(x_len):
        if state_list[i] == 0 and state_list[i+1] == 2: #找到所有0-2组合并记录0对应的位置
            x2_position.append(i)
        if state_list[i] == 2 and state_list[i+1] == 0: #找到所有2-0组合并记录0对应的位置
            y2_position.append(i+1)
        if state_list[i] == 0 and state_list[i+1] == 1: #找到所有0-1组合并记录0对应的位置
            x1_position.append(i)
        if state_list[i] == 1 and state_list[i+1] == 0: #找到所有1-0组合并记录0对应的位置
            y1_position.append(i+1)
    seq2_position = []
    for x in x2_position: #遍历x_position
        for y in y2_position:# 遍历y_position
            if x+2 <= y:
                seq2_position.append([x, x+1, y-1, y])
                break
    seq1_position = []
    for x in x1_position: #遍历x_position
        for y in y1_position:# 遍历y_position
            if x+2 <= y:
                seq1_position.append([x, x+1, y-1, y])
                break
    return seq2_position, seq1_position

def find_valid_sequence(pro_info ,seq_position, valid_threshold=50, invalid_threshold=60):
    """
    根据seq_position中的位置找到符合条件的序列
    条件是0-1-0序列对应的pro_info中的data_num大于设定valid_threshold，
    0-1.。。1-0区间的data_num要小于给定invalid_threshold
    """
    if len(seq_position) == 0:
        return []
    else:
        new_seq_pos = []
        data_num = pro_info.loc[:,'data_num']
        valid = False
        for seq in seq_position:
            if data_num.iloc[seq[0]] > valid_threshold and data_num.iloc[seq[-1]] > valid_threshold and\
                data_num.iloc[seq[1]] > valid_threshold and data_num.iloc[seq[-2]] > valid_threshold:
                valid = True
                if seq[1] != seq[-2]:
                    for seq0 in range(seq[1]+1, seq[-1]):
                        if data_num[seq0] > invalid_threshold:
                            valid = False
                            break
                if valid:
                    new_seq_pos.append(seq)
    return new_seq_pos

def find_va(para_dict, mode, table_name, pro_info, seq, condition='end_time', t_keywords='stime', keywords='voltage'):
    ta = pro_info[condition].iloc[seq[0]] #第一阶段最后一条数据对应的时刻
    #str_value = ta.strftime("%Y-%m-%d %H:%M:%S")
    str_value = str(ta)
    data = rwd.get_bat_1_data(para_dict, mode, table_name, t_keywords, str_value)
    va = data[keywords].iloc[0]
    ta = datetime.datetime.strptime(str_value, "%Y-%m-%d %H:%M:%S")
    return va, ta
    
def find_vb1(para_dict, mode, table_name, pro_info, seq, condition='end_time', t_keywords='stime', keywords='voltage'):
    tb1 = pro_info[condition].iloc[seq[-2]] #倒数第二阶段最后一条数据对应的时刻
    #str_value = tb1.strftime("%Y-%m-%d %H:%M:%S")
    str_value = str(tb1)
    data = rwd.get_bat_1_data(para_dict, mode, table_name, t_keywords, str_value)
    vb1 = data[keywords].iloc[0]
    tb1 = datetime.datetime.strptime(str_value, "%Y-%m-%d %H:%M:%S")
    return vb1, tb1

def find_vb2(para_dict, mode, table_name, pro_info, seq, condition='start_time', t_keywords='stime', keywords='voltage'):
    tb2 = pro_info[condition].iloc[seq[-1]] #倒数第一阶【-1】段第一条数start_time对应的时刻
    #str_value = tb2.strftime("%Y-%m-%d %H:%M:%S")
    str_value = str(tb2)
    data = rwd.get_bat_1_data(para_dict, mode, table_name, t_keywords, str_value)
    vb2 = data[keywords].iloc[0]
    tb2 = datetime.datetime.strptime(str_value, "%Y-%m-%d %H:%M:%S")
    return vb2, tb2

def find_vc(para_dict, mode, table_name, pro_info, seq, state=2, condition1='end_time', condition2='end_time', t_keywords='stime', keywords='voltage'):
    str_value1 = pro_info[condition1].iloc[seq[-1]] 
    str_value1 = str_value1.strftime("%Y-%m-%d %H:%M:%S")
    str_value2 = pro_info[condition2].iloc[seq[-1]] 
    str_value2 = str_value2.strftime("%Y-%m-%d %H:%M:%S")
    df = rwd.read_bat_data(para_dict, mode, table_name, start_time=str_value1, end_time=str_value2)
    if state == 2:
        vc = df[keywords].max()
    else:
        vc = df[keywords].min()
    temp = df[df[keywords] == vc]
    tc = temp[t_keywords].iloc[0]
    #tc = tc.strftime("%Y-%m-%d %H:%M:%S")
    tc = str(tc)
    tc = datetime.datetime.strptime(tc, "%Y-%m-%d %H:%M:%S")
    del df
    return vc, tc

def find_vc_1(para_dict, mode, table_name, pro_info, seq, condition='end_time', t_keywords='stime', keywords='voltage'):
    tc = pro_info[condition].iloc[seq[-1]] #由于静置电压随时间越来越趋于稳定，因此找到此阶段最后时刻的电压即可认为是vc，但是tc是不对的
    str_value = str(tc)
    data = rwd.get_bat_1_data(para_dict, mode, table_name, t_keywords, str_value)
    vc = data[keywords].iloc[0]
    tc = datetime.datetime.strptime(str_value, "%Y-%m-%d %H:%M:%S")
    return vc, tc

def get_feature(para_dict, mode, bat_name, pro_info, keywords='voltage'):
    """
    找到相关的特征参数：先找在第一个静置0最后的电压va，再找放电1倒数第一个电压vb1，和静置0第一个电压vb2,以及对应的时刻tb1，tb2；
    找到最后一个静置0电压不再变化的对应电压值vc以及对应的时刻tc,可认为是当前阶段最高电压
    ir_v_d = va-vb2
    re_v_d = vb2 - vb1
    delta_t = tc-tb2
    充电也一样，但是vc对应的电压值为当前阶段最低电压
    """
    seqs2, seqs1 = find_r_d_sequence(pro_info.loc[:, 'state'].tolist())
    seq2_pos = find_valid_sequence(pro_info, seqs2)
    seq1_pos = find_valid_sequence(pro_info, seqs1)
    feature_list = []
    state = 2 #放电
    for seq in seq2_pos:
        va, ta = find_va(para_dict, mode, bat_name, pro_info, seq)
        vb1, tb1 = find_vb1(para_dict, mode, bat_name, pro_info, seq)
        vb2, tb2 = find_vb2(para_dict, mode, bat_name, pro_info, seq)
        #vc, tc = find_vc(para_dict, mode, bat_name, pro_info, seq, state)
        vc, tc = find_vc_1(para_dict, mode, bat_name, pro_info, seq)#find_vc_1更快
        ir_v_d = va - vb2
        re_v_d = vb2 - vb1
        delta_t = (tc -tb2).seconds
        feature_list.append((state, ir_v_d, re_v_d, delta_t, va, vb1, vb2, vc, ta, tb1, tb2, tc))
    state = 1 #充电
    for seq in seq1_pos:
        va, ta = find_va(para_dict, mode, bat_name, pro_info, seq)
        vb1, tb1 = find_vb1(para_dict, mode, bat_name, pro_info, seq)
        vb2, tb2 = find_vb2(para_dict, mode, bat_name, pro_info, seq)
        #vc, tc = find_vc(para_dict, mode, bat_name, pro_info, seq, state)
        vc, tc = find_vc_1(para_dict, mode, bat_name, pro_info, seq)
        ir_v_d = va - vb2
        re_v_d = vb2 - vb1
        delta_t = (tc -tb2).seconds
        feature_list.append((state, ir_v_d, re_v_d, delta_t, va, vb1, vb2, vc, ta, tb1, tb2, tc))
    feature = pd.DataFrame(feature_list, 
                           columns=('state', 'ir_v_d', 're_v_d', 'delta_t', 'va', 'vb1', 'vb2', 'vc', 'ta', 'tb1', 'tb2', 'tc'))
    return feature
        
def test():
    import pandas as pd
    state_list = [1,2,3,0,1,0,4,1,0,1,1,3,1,0,3,0,1,0]
    data_num_list = [111,222,333,444,555,666,777,888,999,123,234,345,456,567,678,789,890,900]
    df = pd.DataFrame({'state': state_list, 'data_num': data_num_list})
    l = df.loc[:,'state']
    seqs2, seqs1 = find_r_d_sequence(l)
    print(seqs1)
    new_seqs = find_valid_sequence(df, seqs1, 500)
    print(new_seqs)
if __name__ == '__main__':
    test()
