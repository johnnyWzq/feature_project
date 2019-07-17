 #!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Jun 14 15:31:27 2019

@author: wuzhiqiang
提取同一时长下相同电流点来的电压变化特征，供内短路分析用

"""
import pandas as pd
import rw_bat_data as rwd
import lib_path
import g_function as gf

def filter_data(data, max_border, min_border, err_interval, start_border='min'):
    """
    给定一组数据和容忍误差范围，所有在边界内的数值全部改写为边界值
    默认最小边界以最小的数为基准，
    如：data：[17.9，18.1, 15.9, 20.1, 23.1,18.1，-2.1],err=1.6
    max：return:[16.7, 16.7, 15.1, 19.9, 23.1, 16.7, -2.5]
    min：return:[18.7, 18.7, 17.1, 20.3, 23.5, 18.7, -2.1]
    """
    if len(data) <= 1:
        return data
    err_interval = abs(err_interval)
    if start_border == 'min':
        first_border = min_border
        end_border = max_border + err_interval
    elif start_border == 'max':
        first_border = 0 - max_border
        end_border = err_interval - min_border
        
    border_seq = []
    border = first_border
    while border < end_border:
        border_seq.append(round(border, 1))
        border += err_interval
    if start_border == 'max':
        border_seq = [0 - x for x in border_seq]
    new_data = []
    for data0 in data:
        new_data.append(shoot(data0, border_seq, start_border))
    return new_data

def shoot(data0, border_seq, start_border='min'):
    length = int(round(len(border_seq) / 2, 0))
    if start_border == 'min':
        if data0 <= border_seq[0] or len(border_seq) == 1:
            return border_seq[0]
        else:
            border_seq1 = border_seq[:length]
            border_seq2 = border_seq[length:]
            if data0 <= border_seq1[-1]:
                data = shoot(data0, border_seq1)
            else:
                data = shoot(data0, border_seq2)
    else:
        if data0 >= border_seq[0]:
            return border_seq[0]
        else:
            border_seq1 = border_seq[:length]
            border_seq2 = border_seq[length:]
            if data0 >= border_seq1[-1]:
                data = shoot(data0, border_seq1, start_border)
            else:
                data = shoot(data0, border_seq2, start_border)
    return data

def get_process_data(para_dict, mode, table_name, pro_info, process_no, condition1='start_time', condition2='end_time'):
    str_value1 = pro_info[pro_info['process_no'] == process_no][condition1].iloc[0] 
    str_value1 = str_value1.strftime("%Y-%m-%d %H:%M:%S")
    str_value2 = pro_info[pro_info['process_no'] == process_no][condition2].iloc[0] 
    str_value2 = str_value2.strftime("%Y-%m-%d %H:%M:%S")
    df = rwd.read_bat_data(para_dict, mode, table_name, start_time=str_value1, end_time=str_value2)
    return df

def get_appoint_section(data, state, bat_type='NCM', method='soc', keywords='soc', min_len=10):
    """
    获得指定区域的数据，一般采用soc在【40，80】之间的数据
    当数据没有soc时，考虑时间或者电压来判断
    电压：三元 充电3.65-3.95，放电3.55-3.85 铁锂 充电3.4-3.55, 放电3.1-3.2
    时间：按0-100%平均分配时长
    """
    if data is None:
        return None
    if method == 'soc':
        min_border = 0.4
        max_border = 0.8
        data = data[data[keywords] >= min_border][data[keywords] <= max_border]
    elif method == 'voltage':
        conditions_dict = {'NCM': {'charge': [3.65, 3.95], 'discharge': [3.55, 3.85]},
                           'LFP': {'charge': [3.4, 3.55], 'discharge': [3.1, 3.2]}
                           }
        if bat_type in conditions_dict.keys():
            min_border = conditions_dict[bat_type][state][0]
            max_border = conditions_dict[bat_type][state][1]
            data = data[data[keywords] >= min_border][data[keywords] <= max_border]
        else:
            print('the operation is failure because the battery type is not in the config.')
    elif method == 'tick':
        min_border = int(0.4 * len(data))
        max_border = int(0.8 * len(data))
        data = data[min_border: max_border]
    if len(data) <= min_len:
        print('the data is lack of calculating.')
        return None
    else:
        return data
    
def get_same_current(data, keywords='current_reg', min_len=10, min_cur=10):
    #获得对应的index,每个cur仅取最长的那部分数据
    if data is None:
        return None
    pro_data = pd.DataFrame()
    data_gp = data.groupby(keywords)
    for key in data_gp.groups.keys():
        df = data_gp.get_group(key)
        if len(df) <= 1:
            continue
        start = 0
        clip_list = []
        for i in range(1, len(df)):
            if df.index[i] - df.index[i-1] != 1 or i == (len(df) - 1):
                if i == (len(df) - 1):
                    clip_list.append([df.index[start], df.index[i]])
                else:
                    clip_list.append([df.index[start], df.index[i - 1]])
                    start = i
        res = clip_list[0]
        if len(clip_list) > 1:
            clip_len = res[1] - res[0]
            for clip in clip_list:
                if clip[1] - clip[0] >= clip_len:
                    res = clip
        if (res[1] - res[0]) >= min_len and abs(key) >= min_cur:
            pro_data = pro_data.append(df.loc[res[0]: res[1]])
            del df
    return pro_data

def get_same_current_process(data, c_keywords='current_reg', p_keywords='process_no', v_keywords='voltage'):
    if len(data) == 0:
        return None
    res = pd.DataFrame()
    tick = 0
    data_gp = data.groupby(c_keywords)
    for cur in data_gp.groups.keys(): #按电流分组
        df = data_gp.get_group(cur)
        df_gp = df.groupby(p_keywords)
        #获得相同电流的各过程数据，但是还需要找到同等时长
        duration = {'duration': [], 'process_no':[]}
        if len(df_gp.groups.keys()) > 1: #有1个以上的过程有相同电流
            for process_no in df_gp.groups.keys():#按过程分组
                df0 = df_gp.get_group(process_no) #获得具有相同电流的不同过程数据
                duration['duration'].append(len(df0))
                duration['process_no'].append(process_no)
            duration = pd.DataFrame.from_dict(duration)
            if len(duration) > 2:
                sel_duration = duration['duration'].median()
                duration = duration[duration['duration'] >= sel_duration]
            sel_duration = duration['duration'].min() #默认取时间最短的作为交集
            
            for process_no in duration['process_no']:#重新分组，取相同时间的数据,假设电压的变化是线性的，取数从位置0开始
                df0 = df_gp.get_group(process_no) #获得具有相同电流的不同过程数据
                df0 = df0[:sel_duration]
                
                df1 = pd.DataFrame(columns=data.columns)
                for column in df1.columns:
                    df1.loc[tick, column] = float(df0[column].iloc[0])
                df1.loc[tick, 'delta_voltage'] = df0[v_keywords].iloc[-1] - df0[v_keywords].iloc[0]
                df1.loc[tick, 'duration'] = sel_duration
                df1 = gf.cal_stat_row(tick, df0[v_keywords], v_keywords, df1)
                df1 = gf.cal_stat_row(tick, df0[v_keywords].diff(), v_keywords + '_diff', df1)
                df1 = gf.cal_stat_row(tick, df0[v_keywords].diff().diff(), v_keywords + '_diff2', df1)
                df1 = gf.cal_stat_row(tick, df0[v_keywords].diff() / df0[v_keywords], v_keywords + '_diffrate', df1)
                
                res = res.append(df1)
                del df1
                tick += 1
    return res
                
def get_feature_stdv(para_dict, mode, bat_name, pro_info, v_keywords='voltage', c_keywords='current'):
    """
    找到相关的特征参数：先将所有过程按充放静置状态进行分离和过滤，将符合条件的过程编号process_no放入dict中
    将所有电流进行模拟滤波filter_data，以便减小电流值分布，设置最大最小值为2C倍率电流值
    按充电和放电过程分别处理，充电：
    将所有充电
    充电也一样，但是vc对应的电压值为当前阶段最低电压
    """
    process_no_dict = gf.filter_sequence(pro_info, r_filter=100, c_filter=100, d_filter=80,)#获得所需各阶段数据
    if process_no_dict is None:
        print('there is no data can be matched.')
        return None
    C_RATE = para_dict['bat_config']['C_RATE']
    bat_type = para_dict['bat_config']['bat_type']
    border_dict = {'rest':[-round(C_RATE/10, 1), round(C_RATE/10, 1)],
                           'charge':[-3 * C_RATE, 2 * C_RATE],
                          'discharge':[-C_RATE * 3, 2 * C_RATE]}
    feature = pd.DataFrame()
    for key, process_no_list in process_no_dict.items():
        if key == 'rest':
            continue
        data = []
        for process_no in process_no_list:
            process_data = get_process_data(para_dict, mode, bat_name, pro_info, process_no)
            process_data = get_appoint_section(process_data, key, bat_type, method='voltage', keywords='voltage')
            if process_data is not None:
                process_data = process_data[[v_keywords, c_keywords]]
                process_data['current_reg'] = filter_data(process_data[c_keywords].tolist(),
                                        border_dict[key][1], border_dict[key][0], border_dict['rest'][1])
                process_data['process_no'] = process_no
                process_data = get_same_current(process_data, min_cur=1)#, min_cur=int(C_RATE/3))
                data.append(process_data)
        data = pd.concat(tuple(data)) #获得清理过的数据，包含符合条件的过程数据
        data = get_same_current_process(data)
        if data is not None:
            feature = feature.append(data)
            del data
    if len(feature) != 0:
        feature = feature.sort_values('process_no')
        feature = feature.reset_index(drop=True)
    return feature

def test():
    a = [17.9, 18.1, 15.9, 20.1, 23.1, 18.1, -2.1]
    min_border = min(a)
    max_border = max(a)
    b = filter_data(a, max_border, min_border,1.6, 'min')
    print(a)
    print(b)

if __name__ == '__main__':
    test()
    