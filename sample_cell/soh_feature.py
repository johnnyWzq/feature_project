#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Jul  8 11:43:44 2019

@author: wuzhiqiang
#1峰soc大概在40%，2峰大概在80%，1-2间谷大概60%
"""
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import leastsq
import rw_bat_data as rwd

def slip_by_turing_point(data, direction=0): #未使用
    df_list = []
    if direction == 0:
        df = data[data['diff'] <= 0]#获得下降的点
    else:
        df = data[data['diff'] >= 0]#获得下降的点
    if df.empty:
        df_list.append(data)
    else:
        index_list = df.index.tolist()
        if index_list[-1] < data.index[-1]:
            index_list.append(data.index[-1])
        start = 0
        for index in index_list:
            df_list.append(data.loc[start: index-1])
            start = index
    return df_list

def analysis_(total_data, peak_2_pos, state):
    df1 = total_data.loc[:peak_2_pos, ]
    #df1.reverse()
    length1 = len(df1)
    df2 = total_data.loc[peak_2_pos:, ]
    length2 = len(df2)
    #del dqdv_list
    #进行曲线分析
    #fit_pars_1 = analysis_dqdv_curve(df1, i, pro_info, 6)
    #fit_pars_2 = analysis_dqdv_curve(df2, i, pro_info, 3)
    #fit_pars_3 = analysis_dqdv_curve(df1.loc[:length1//2, ], i, pro_info, 6)
    #fit_pars_4 = analysis_dqdv_curve(df1.loc[length1//2:, ], i, pro_info, 6)
    #fit_pars_5 = analysis_dqdv_curve(df2.loc[:length2//2, ], i, pro_info, 3)
    #fit_pars_6 = analysis_dqdv_curve(df2.loc[length2//2:, ], i, pro_info, 3)
    
    #temp1.append(fit_pars_1)
    #temp1.append(fit_pars_3)
    #temp1.append(fit_pars_4)
    #temp2.append(fit_pars_2)
    #temp2.append(fit_pars_5)
    #temp2.append(fit_pars_6)
    #获得各曲线斜率
    find_1st_peak(df1, state, p=6)
    #find_1st_peak(df1.iloc[:length1//2, ].copy(), state, p=5)
    find_1st_peak(df1.iloc[length1//2:, ], state, p=5)
    find_3rd_peak(df2, state, p=3)
    find_3rd_peak(df2.iloc[:length2//2, ].copy(), state, p=3)
    #find_3rd_peak(df2.iloc[length2//2:, ], state, p=3)
            
def analysis_dqdv_curve(df, i, pro_info, p):
    import matplotlib.pyplot as plt
    #df = pd.DataFrame(dqdv_list, columns=['dqdv'])
    plt.figure()
    #plt.plot(df['dqdv'])
    #plt.show()
    print(pro_info['state'].iloc[i])
    x_= df.index
    y_= df['dqdv']
    X = x_
    Y = y_
    #print(fitting(3, data.index, data['dqdv']))
    fit_pars = fitting(p, X, Y)[0]
    plt.plot(x_, y_, label='real line')
    plt.scatter(X, Y, label='real points')
    plt.plot(x_, np.poly1d(fit_pars)(x_), label='fitting line')
    plt.legend()
    plt.show()
    print(fit_pars)
    return fit_pars

def analysis_dqdv_curve2(df, state, peak_pos_list, valley_pos_list):
    if state == 1:
        state = 'charge'
    elif state == 2:
        state = 'discharge'
    #x = df.index
    #x = df['x']
    x = df['voltage_mean']
    y1 = df['dqdv']
    y2 = df['dqdv_slm']
    import matplotlib.pyplot as plt
    fig, ax = plt.subplots(figsize=(6, 6))
    ax.grid(linestyle="--", linewidth=0.5, color='.25', zorder=-10)
    ax.plot(x, y1, label='real line')
    ax.scatter(x, y1, label='real points')
    ax.plot(x, y2, label='fitting line')
    ax.set_title(state)
    ax.legend()
    def circle(x, y, radius=0.15, color='black'):
        from matplotlib.patches import Circle
        circle = Circle((x, y), radius, clip_on=False, zorder=10, linewidth=1,
                        edgecolor=color, facecolor=(0, 0, 0, .0125))
        ax.add_artist(circle)
        
    for peak_pos in peak_pos_list:
        #circle(peak_pos, df['dqdv'].loc[peak_pos])
        plt.scatter(df['voltage_mean'].loc[peak_pos], df['dqdv'].loc[peak_pos], color='', marker='o', edgecolors='r', s=200)
    for valley_pos in valley_pos_list:
        #circle(valley_pos, df['dqdv'].loc[valley_pos], color='red')
        plt.scatter(df['voltage_mean'].loc[valley_pos], df['dqdv'].loc[valley_pos], color='', marker='o', edgecolors='g', s=200)
    plt.show()
    print(peak_pos_list)
    print(valley_pos_list)
    
def generate_train_data(total_data, state, method='voltage'):
    """
    #考虑后期应用数据的质量，从完整的一个过程数据中生成新的数据供训练用
    #采用soc或电压
    #充电：soc取40-85，    50-85,    40-95,      50-90,     75-90;
    #     volt 3.51-3.7  3.59-3.7  3.51-4.05   3.59-4.05  3.64-4.05
    #放电：soc取40-85，    50-85,    40-95,      50-90,     75-90;
    #     volt 3.1-3.55  3.2-3.55  3.1-4.05   3.59-4.05  3.64-4.05
    #key 代表数据筛选条件
    """
    train_data_dict = {'0-100': total_data}
    return train_data_dict
    
def get_1_pro_data(para_dict, mode, table_name, pro_info, pro_no, condition1='start_time', condition2='end_time', t_keywords='stime'):
    str_value1 = pro_info[condition1].iloc[pro_no] 
    str_value1 = str_value1.strftime("%Y-%m-%d %H:%M:%S")
    str_value2 = pro_info[condition2].iloc[pro_no] 
    str_value2 = str_value2.strftime("%Y-%m-%d %H:%M:%S")
    df = rwd.read_bat_data(para_dict, mode, table_name, start_time=str_value1, end_time=str_value2)
    return df

def slip_data_by_volt(df, delta_v=0.01, series=1, keywords='voltage'):
    """
    #按delta_v进行切片，series为电芯串联数
    """
    if df is None or len(df) <= 1:
        print('there is not enough data.')
        return None
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
    return clip_data_list, pos_seq

def calc_other_vectors(df, state):
    """
    """
    if state == 1: #充电
        df['c'] = df['charge_c']
        df['e'] = df['charge_e']
    elif state == 2: #放电
        df['c'] = df['discharge_c']
        df['e'] = df['discharge_e']
    else:
        df['c'] = 0
        df['e'] = 0
    return df

def calc_dqdv(df, bias, C_RATE, parallel=1, sample=1):
    """
    #计算dq/dv，由于没有dq，使用i代替，但需要考虑采样频率换算成1s
    """
    dqdv = (df['current'].sum() / sample) / (df['voltage'].iloc[-1] - df['voltage'].iloc[0])
    dqdv = dqdv / C_RATE / parallel
    if dqdv == np.inf or dqdv == -np.inf: #电压变化较快，一条数据就超过设定值
        dqdv = 88888888
    return dqdv

def outlier_err_dqdv(dqdv_list, method=2):
    """
    #1种是比较平均值找异常
    #2种是比较相邻，相差太大异常
    """
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
    


def scale_curve(df, is_scale=False, max_y=10000.0, min_y=0):
    """
    #将曲线进行标准化
    """
    #平移Y
    scale = 1
    if is_scale:
        df['dqdv_slm'] = df['dqdv_slm'] - df['dqdv_slm'].iloc[0]
        df['dqdv'] = df['dqdv'] - df['dqdv'].iloc[0]
        #缩放Y-X
        scale = df['dqdv_slm'].max() / max_y
        df['dqdv_slm'] = df['dqdv_slm'] / scale
        scale = df['dqdv'].max() / max_y
        df['dqdv'] = df['dqdv'] / scale
        df['x'] = df['x'] / scale
    return df, scale

def sel_curve_para(df, state, scale, direction='left'):
    data_len = len(df)
    if direction == 'left':
        if data_len <= 25:
            rate = 250
            peak_incline = 0.65
            valley_incline = 1.5
        else:
            rate = 100
            peak_incline = 0.76
            valley_incline = 1.2
    elif direction == 'right':
        if data_len <= 25:
            rate = 60
            peak_incline = 0.65
            valley_incline = 0.75
        else:
            rate = 35
            peak_incline = 0.76
            valley_incline = 0.86
    peak_value = rate / scale * peak_incline
    valley_value = rate / scale * valley_incline
    return peak_incline, valley_incline, peak_value, valley_value
    
def find_break_point(data, num, peak_incline, valley_incline, peak_value, valley_value, direction='left'):
    """
    #direction：0 上升， 1 下降
    #当斜率变化超过设定时，认为拐点 |diff(x)/diff(x-1)|<=up_incline 或>= (down_incline)
    #拐点（峰）dff2<=0,且diff(x-1)>=设定,且diff(x+1)/diff(x)<=up_incline [不采用f(x)>=f(x-1)&&f(x)>f(x+1)或者f(x)>f(x-1)&&f(x)>=f(x+1)]
    #拐点(谷) diff2>=0,且diff(x-1)<=设定,且diff(x+1)/diff(x)>=down_incline [不采用f(x)<=f(x-1)&&f(x)<f(x+1)或者f(x)<f(x-1)&&f(x)<=f(x+1)]
    """
    length = len(data)
    if length < num:
        return [], []
    #rate = data['dqdv_slm'].max() / length #y轴刻度转换比例
    #将标准斜率转化为实际的diff
    #incline *= rate
    #up_incline *= rate
    #down_incline *= rate
    #data['dqdv_slm_nml'] = data['dqdv_slm'] / rate #x\y轴尺度一致
    data['dqdv_slm_diff'] = data['dqdv_slm'].diff()
    data['dqdv_slm_diff2'] = data['dqdv_slm_diff'].diff()
    data['dqdv_slm_diffrate'] = 1
    for i in range(2, len(data)):
        data['dqdv_slm_diffrate'].iloc[i] = data['dqdv_slm_diff'].iloc[i] / data['dqdv_slm_diff'].iloc[i - 1]
    data = data.fillna(0)
    #df = data[data['dqdv_slm_diff2'].abs() >= incline].copy() #小于设定
    peak_pos_list_tmp = data[data['dqdv_slm_diffrate'] <= peak_incline].index.tolist()
    valley_pos_list_tmp = data[data['dqdv_slm_diff2'] >= 0].index.tolist()
    #valley_pos_list_tmp = data[data['dqdv_slm_diffrate'] >= valley_incline].index.tolist()
    #df = df[['dqdv_slm_diff2']]
    if (len(peak_pos_list_tmp) + len(valley_pos_list_tmp)) < 1:
        return [], []
    peak_pos_list = []
    valley_pos_list = []
    #peak_pos_list_tmp = df[df['dqdv_slm_diff2'] < 0].index.tolist()
    #valley_pos_list_tmp = df[df['dqdv_slm_diff2'] > 0].index.tolist()
    #进行前一个位置斜率判断
    for peak_pos in peak_pos_list_tmp:
        if peak_pos in data.index[:num] or peak_pos in data.index[-num:]:
            continue
        if data['dqdv_slm_diff'].loc[peak_pos - 1] >= peak_value and\
            data['dqdv_slm_diffrate'].loc[peak_pos + 1] <= peak_incline and\
            data['dqdv_slm_diff2'].loc[peak_pos + 1] <= 0:
            peak_pos_list.append(peak_pos)
    #for valley_pos in valley_pos_list_tmp[:-1]:
    for valley_pos in valley_pos_list_tmp:
        #if data['dqdv_slm_diff'].loc[valley_pos - 1] <= valley_value: and\
            #data['dqdv_slm_diffrate'].loc[valley_pos + 1] >= valley_incline or\
            #data['dqdv_slm_diffrate'].loc[valley_pos + 1] < 0
            #data['dqdv_slm_diff2'].loc[valley_pos + 1] >= 0:
        if valley_pos in data.index[:num] or valley_pos in data.index[-num:]:
            continue
        if (direction == 'left' and data['dqdv_slm_diff'].loc[valley_pos - 1] <= valley_value) or\
            (direction == 'right' and data['dqdv_slm_diffrate'].loc[valley_pos + 1] <= valley_incline and\
             abs(data['dqdv_slm_diff'].loc[valley_pos - 1]) <= valley_value):
            valley_pos_list.append(valley_pos)
    peak_pos_list_tmp = peak_pos_list
    valley_pos_list_tmp = valley_pos_list
    peak_pos_list = []
    valley_pos_list = []
    if len(peak_pos_list_tmp) > 0:#大于空
        peak_pos_list.append(peak_pos_list_tmp[0])
        for i in range(1, len(peak_pos_list_tmp)):
            if (peak_pos_list_tmp[i] - peak_pos_list_tmp[i - 1]) > num:
                peak_pos_list.append(peak_pos_list_tmp[i])#上升取连续位置的开头位置
    if len(valley_pos_list_tmp) > 0:#大于空
        valley_pos_list.append(valley_pos_list_tmp[0])
        for i in range(1, len(valley_pos_list_tmp)):
            if (valley_pos_list_tmp[i] - valley_pos_list_tmp[i - 1]) > num:
                valley_pos_list.append(valley_pos_list_tmp[i])    
    return peak_pos_list, valley_pos_list


def find_1st_peak(df, state, p=6, rate=400, incline=0.3, duration=3):
    """
    #找1峰，由于1峰在2峰左侧，找到2峰后，从2峰左侧位置开始判断
    #1峰判断规则为斜率发生较大变化incline，且斜率需要维持至少duration，并且右斜率小于左斜率，且总体均为正
    """
    df = smooth_curve(df, p)
    df, scale = scale_curve(df, is_scale=False)
    peak_incline, valley_incline, peak_value, valley_value = sel_curve_para(df, state, scale)
    peak_pos_list, valley_pos_list = find_break_point(df, duration, peak_incline, valley_incline, peak_value, valley_value)
    analysis_dqdv_curve2(df, state, peak_pos_list, valley_pos_list)
    peak_pos, valley_pos = get_valid_pos(peak_pos_list, valley_pos_list, state)
    return peak_pos, valley_pos

def find_3rd_peak(df, state, p=6, rate=400, incline=0.3, duration=3):
    """
    """
    df = smooth_curve(df, p)
    df, scale = scale_curve(df, is_scale=False)
    peak_incline, valley_incline, peak_value, valley_value = sel_curve_para(df, state, scale, direction='right')
    peak_pos_list, valley_pos_list = find_break_point(df, duration, peak_incline, valley_incline, peak_value, valley_value, 'right')
    analysis_dqdv_curve2(df, state, peak_pos_list, valley_pos_list)
    peak_pos, valley_pos = get_valid_pos(peak_pos_list, valley_pos_list, state)
    return peak_pos, valley_pos

def find_2nd_peak(data, duration=3):
    """
    #找到ic曲线的第2峰
    2峰为dqdv最大值且不在起始或终止位置
    """
    #df = df.reset_index(drop=True)
    temp_value = max(data)#temp_value = data.max()
    temp_pos = data.index(temp_value)#data.idxmax(skipna=True)
    length = len(data)
    peak_2 = None
    peak_2_pos = None
    if (temp_pos + duration) < length and (temp_pos - duration) > 0:
        peak_2 = temp_value
        peak_2_pos = temp_pos
    return peak_2, peak_2_pos

def get_valid_pos(peak_pos_list, valley_pos_list, state):
    peak_pos, valley_pos = 0, 0 #默认为没有拐点
    if state == 1:
        if len(peak_pos_list) > 0:
            peak_pos = min(peak_pos_list)
        if len(valley_pos_list) > 0:
            valley_pos = max(valley_pos_list)
    elif state == 2:
        if len(peak_pos_list) > 0:
            peak_pos = max(peak_pos_list)
        if len(valley_pos_list) > 0:
            valley_pos = min(valley_pos_list)
    return peak_pos, valley_pos

def find_ic_feature(df, state, C_RATE):
    """
    #先找第2峰，如果存在，则继续找1峰和3峰
    #特征包括：1、2、3峰值和位置（对应电压），2峰两边斜率，2峰面积
    """
    clip_data_list, pos_seq = slip_data_by_volt(df)
    total_data = pd.DataFrame()
    dqdv_list = []
    feature_cols =  ['voltage_', 'dqdv', 'peak_incline', 'valley_incline']
    j = 0
    for clip_data in clip_data_list:
        dqdv = calc_dqdv(clip_data, 0, C_RATE)
        if dqdv != 88888888:
            dqdv_list.append(dqdv)
            total_data = total_data.append(rwd.transfer_data(j, clip_data, keywords='stime')) #获得每一个电压数据片对电压的统计值
            j += 1 #只有当dqdv有效，j才增加
    del clip_data_list
    dqdv_list = outlier_err_dqdv(dqdv_list)#对异常点进行替换
    total_data['dqdv'] = dqdv_list
    peak_2, peak_2_pos = find_2nd_peak(dqdv_list)#获得2峰，并将数据分为2部分,保留原index
    if peak_2_pos is not None:
        #analysis_(total_data, peak_2_pos, state)#分析
        feature_df = pd.DataFrame()  #创建保存特征的数据块
        feature_2 = total_data.loc[[peak_2_pos]].copy()
        y2 = peak_2
        x2 = feature_2['voltage_mean'].loc[peak_2_pos]
        
        feature_2 = change_columns(feature_2, '_2', *feature_cols)
        feature_df = feature_df.append(feature_2) #将2peak点的特征放入
        feature_df = feature_df.reset_index(drop=True)
        
        peak_1_pos_p, peak_1_pos_v = find_1st_peak(total_data[:peak_2_pos], state, p=6)
        feature_1_peak = total_data.loc[[peak_1_pos_p]].copy()
        y1 = feature_1_peak['dqdv'].loc[peak_1_pos_p]
        x1 = feature_1_peak['voltage_mean'].loc[peak_1_pos_p]
        feature_1_peak['peak_incline'] = (y2 - y1) / (x2 - x1)
        feature_1_peak = feature_1_peak.reset_index(drop=True)
        if peak_1_pos_p == 0:#无拐点
            feature_1_peak.loc[:] = -99999999 #
        feature_1_peak = change_columns(feature_1_peak, '_p', *feature_cols)
        
        feature_1_valley = total_data.loc[[peak_1_pos_v]].copy()
        y1 = feature_1_valley['dqdv'].loc[peak_1_pos_v]
        x1 = feature_1_valley['voltage_mean'].loc[peak_1_pos_v]
        feature_1_valley['valley_incline'] = (y2 - y1) / (x2 - x1)
        feature_1_valley = feature_1_valley.reset_index(drop=True)
        if peak_1_pos_v == 0:
            feature_1_valley.loc[:] = -99999999
        feature_1_valley = change_columns(feature_1_valley, '_v', *feature_cols)
        
        feature_1_peak = feature_1_peak.merge(feature_1_valley, left_index=True, right_index=True)
        feature_1_peak = change_columns(feature_1_peak, '_1', *feature_cols)
        feature_df = feature_df.merge(feature_1_peak, left_index=True, right_index=True)
        del feature_1_peak
        del feature_1_valley
        
        peak_3_pos_p, peak_3_pos_v = find_3rd_peak(total_data[peak_2_pos:], state, p=3)
        feature_3_peak = total_data.loc[[peak_3_pos_p]].copy()
        y1 = feature_3_peak['dqdv'].loc[peak_3_pos_p]
        x1 = feature_3_peak['voltage_mean'].loc[peak_3_pos_p]
        feature_3_peak['peak_incline'] = (y2 - y1) / (x2 - x1)
        feature_3_peak = feature_3_peak.reset_index(drop=True)
        if peak_3_pos_p == 0:#无拐点
            feature_3_peak.loc[:] = -99999999 #
        feature_3_peak = change_columns(feature_3_peak, '_p', *feature_cols)
        
        feature_3_valley = total_data.loc[[peak_3_pos_v]].copy()
        y1 = feature_3_valley['dqdv'].loc[peak_3_pos_v]
        x1 = feature_3_valley['voltage_mean'].loc[peak_3_pos_v]
        feature_3_valley['valley_incline'] = (y2 - y1) / (x2 - x1)
        feature_3_valley = feature_3_valley.reset_index(drop=True)
        if peak_3_pos_v == 0:
            feature_3_valley.loc[:] = -99999999
        feature_3_valley = change_columns(feature_3_valley, '_v', *feature_cols)
        
        feature_3_peak = feature_3_peak.merge(feature_3_valley, left_index=True, right_index=True)
        feature_3_peak = change_columns(feature_3_peak, '_3', *feature_cols)
        feature_df = feature_df.merge(feature_3_peak, left_index=True, right_index=True)
        del feature_3_peak
        del feature_3_valley
        return feature_df
    else:
        print('the data can not be used.')
        return None
    
def change_columns(data, col, *kwg):
    col_names = []
    for i in data.columns:
        for j in kwg:
            if j in i:
                col_names.append(i)
                break
    col_list = []
    for i in col_names:
        tmp = data[i]
        data[i + str(col)] = tmp
        col_list.append(i + str(col))
    return data[col_list]

def get_feature_soh(para_dict, mode, bat_name, pro_info, keywords='voltage'):
    pro_info = pro_info[pro_info['state'] != 0]
    if len(pro_info) < 1:
        return None
    C_RATE = para_dict['bat_config']['C_RATE']
    #V_RATE = para_dict['bat_config']['V_RATE']
    #bat_type = para_dict['bat_config']['bat_type']
    train_feature = []
    for i in range(0, 100):#range(len(pro_info)):
        print('starting calculating the features of battery for soh...')
        state = pro_info['state'].iloc[i]
        df = get_1_pro_data(para_dict, mode, bat_name, pro_info, i)
        df = df.reset_index(drop=True)
        df = calc_other_vectors(df, state)
        cycle_soh = df['c'].iloc[-1] / C_RATE
        train_data_dict = generate_train_data(df, state)#生产新的训练数据
        for key, train_data in train_data_dict.items():
            feature_df = find_ic_feature(train_data, state, C_RATE)
            feature_df['state'] = state
            feature_df['section'] = key
            feature_df['process_no'] = pro_info['process_no'].iloc[i]
            feature_df['soh'] = cycle_soh
        train_feature.append(feature_df)
        del train_data_dict
    train_feature = pd.concat(tuple(train_feature))
    rwd.save_train_xlsx(train_feature, 'cell_soh_'+bat_name, './data/')
    return train_feature


# 误差函数， 计算拟合曲线与真实数据点之间的差 ，作为leastsq函数的输入
def residuals(p, x, y):
    fun = np.poly1d(p)    # poly1d（）函数可以按照输入的列表p返回一个多项式函数
    return y - fun(x)

# 拟合函数
def fitting(p, X, Y):
    pars = np.random.rand(p+1)  # 生成p+1个随机数的列表，这样poly1d函数返回的多项式次数就是p
    r = leastsq(residuals, pars, args=(X, Y))   # 三个参数：误差函数、函数参数列表、数据点
    return r

def smooth_curve(df, p):
    X = df.index
    Y = df['dqdv']
    fit_pars = fitting(p, X, Y)[0]
    df['dqdv_slm'] = np.poly1d(fit_pars)(X)
    df['x'] = df.index
    return df

def test():
    data = [1,2,3,4,5,4,5,7,9,11,13,9.5,10,10.5,11,11,10.5,11,13,14,15]
    #data.reverse()
    data = pd.DataFrame(data, columns=['dqdv'])
    #df_list = find_1st_peak(data)
    x_= data.index
    y_= data['dqdv']
    X = x_
    Y = y_
    #print(fitting(3, data.index, data['dqdv']))
    fit_pars = fitting(4, X, Y)[0]
    plt.plot(x_, y_, label='real line')
    plt.scatter(X, Y, label='real points')
    plt.plot(x_, np.poly1d(fit_pars)(x_), label='fitting line')
    plt.legend()
    plt.show()
    print(np.poly1d(fit_pars)(x_))
    data['dqdv_slm'] = np.poly1d(fit_pars)(x_)
    peak_pos_list, valley_pos_list = find_break_point(data, 3, 0.25)
    print(peak_pos_list)
    print(valley_pos_list)
if __name__ == '__main__':
    test()