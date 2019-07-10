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

def calc_dqdv(df, bias, C_RATE, sample=1):
    """
    #计算dq/dv，由于没有dq，使用i代替，但需要考虑采样频率换算成1s
    """
    dqdv = (df['current'].sum() / sample) / (df['voltage'].iloc[-1] - df['voltage'].iloc[0])
    dqdv = dqdv / C_RATE
    if dqdv == np.inf or dqdv == -np.inf: #电压变化较快，一条数据就超过设定值
        dqdv = 88888888
    return dqdv
    
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
    x = df.index
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
        plt.scatter(peak_pos, df['dqdv'].loc[peak_pos], color='', marker='o', edgecolors='g', s=200)
    for valley_pos in valley_pos_list:
        #circle(valley_pos, df['dqdv'].loc[valley_pos], color='red')
        plt.scatter(valley_pos, df['dqdv'].loc[valley_pos], color='', marker='o', edgecolors='r', s=200)
    plt.show()
    print(peak_pos_list)
    print(valley_pos_list)
    
    
def find_ic_feature():
    """
    #先找第2峰，如果存在，则继续找1峰和3峰
    #特征包括：1、2、3峰值和位置（对应电压），2峰两边斜率，2峰面积
    """
    
def find_break_point(data, num, incline=0.3, up_incline=0.1, down_incline=-0.1):
    """
    #direction：0 上升， 1 下降
    #当斜率变化超过设定时，认为拐点 |diff(x)-diff(x-1)|>=incline
    #拐点（峰）dff2<0,且diff>=设定, [不采用f(x)>=f(x-1)&&f(x)>f(x+1)或者f(x)>f(x-1)&&f(x)>=f(x+1)]
    #拐点(谷) diff2>0,且diff<=设定, [不采用f(x)<=f(x-1)&&f(x)<f(x+1)或者f(x)<f(x-1)&&f(x)<=f(x+1)]
    """
    length = len(data)
    if length < num:
        return [], []
    rate = data['dqdv_slm'].max() / length #y轴刻度转换比例
    data['dqdv_slm_nml'] = data['dqdv_slm'] / rate #x\y轴尺度一致
    data['dqdv_slm_diff'] = data['dqdv_slm_nml'].diff()
    data['dqdv_slm_diff'].loc[0] = 0
    data['dqdv_slm_diff2'] = data['dqdv_slm_diff'].diff()
    df = data[data['dqdv_slm_diff2'].abs() >= incline].copy() #取斜率大于设定
    df = df[['dqdv_slm_diff2']]
    if len(df) < 1:
        return [], []
    peak_pos_list = []
    valley_pos_list = []
    peak_pos_list_tmp = df[df['dqdv_slm_diff2'] < 0].index.tolist()
    valley_pos_list_tmp = df[df['dqdv_slm_diff2'] > 0].index.tolist()
    for peak_pos in peak_pos_list_tmp:
        if data['dqdv_slm_diff'].loc[peak_pos] >= up_incline:
            peak_pos_list_tmp.remove(peak_pos)
    for valley_pos in valley_pos_list_tmp:
        if data['dqdv_slm_diff'].loc[valley_pos] <= down_incline:
            valley_pos_list_tmp.remove(valley_pos)
    if len(peak_pos_list_tmp) > 1:#大于空或者1
        peak_pos_list.append(peak_pos_list_tmp[0])
        for i in range(1, len(peak_pos_list_tmp)):
            if (peak_pos_list_tmp[i] - peak_pos_list_tmp[i - 1]) > num:
                peak_pos_list.append(peak_pos_list_tmp[i])#上升取连续位置的开头位置
    if len(valley_pos_list_tmp) > 1:#大于空或者1
        valley_pos_list.append(valley_pos_list_tmp[0])
        for i in range(1, len(valley_pos_list_tmp)):
            if (valley_pos_list_tmp[i] - valley_pos_list_tmp[i - 1]) > num:
                valley_pos_list.append(valley_pos_list_tmp[i])    
    return peak_pos_list, valley_pos_list

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
    
    
def find_1st_peak(df, state, p=6, incline=0.3, duration=3):
    """
    #找1峰，由于1峰在2峰左侧，找到2峰后，从2峰左侧位置开始判断
    #1峰判断规则为斜率发生较大变化incline，且斜率需要维持至少duration，并且右斜率小于左斜率，且总体均为正
    """
    df = smooth_curve(df, p)
    peak_pos_list, valley_pos_list = find_break_point(df, duration, incline)
    analysis_dqdv_curve2(df, state, peak_pos_list, valley_pos_list)

def find_3rd_peak(df, incline=0.3, duration=3):
    """
    """
    df['diff'] = df['dqdv'].diff()
    df['diff2'] = df['diff'].diff()
    temp = df[df['diff2'].abs() > incline] #斜率变化大于设定
    cnt = len(temp)
    if cnt <= 2:#没有1峰
        return None
    df_list = find_break_point(df, duration)
    return df_list
        
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

def get_feature_soh(para_dict, mode, bat_name, pro_info, keywords='voltage'):
    pro_info = pro_info[pro_info['state'] != 0]
    if len(pro_info) < 1:
        return None
    C_RATE = para_dict['bat_config']['C_RATE']
    #V_RATE = para_dict['bat_config']['V_RATE']
    #bat_type = para_dict['bat_config']['bat_type']
    fit_pars_list = []
    temp1 = []
    temp2 = []
    for i in range(18, 20):#range(len(pro_info)):
        state = pro_info['state'].iloc[i]
        df = get_1_pro_data(para_dict, mode, bat_name, pro_info, i)
        df = df.reset_index(drop=True)
        clip_data_list, pos_seq = slip_data_by_volt(df)
        dqdv_list = []
        for clip_data in clip_data_list:
            dqdv = calc_dqdv(clip_data, 0, C_RATE)
            if dqdv != 88888888:
                dqdv_list.append(dqdv)
        dqdv_list = outlier_err_dqdv(dqdv_list)#对异常点进行替换
        peak_2, peak_2_pos = find_2nd_peak(dqdv_list)#获得2峰，并将数据分为2部分,保留原index
        if peak_2_pos is not None:
            df1 = dqdv_list[:peak_2_pos]
            #df1.reverse()
            df1 = pd.DataFrame(df1, columns=['dqdv'])
            length1 = len(df1)
            df2 = dqdv_list[peak_2_pos:]
            df2 = pd.DataFrame(df2, columns=['dqdv'])
            length2 = len(df2)
            del dqdv_list
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
            find_1st_peak(df1, state)
            find_1st_peak(df1.loc[:length1//2, ], state)
            find_1st_peak(df1.loc[length1//2:, ], state)
        else:
            print('the data can not be used.')
    #fit_pars_ = pd.DataFrame(fit_pars_list)
    #fit_pars_ = fit_pars_.round(6)
    temp1 = pd.DataFrame(temp1)
    temp2 = pd.DataFrame(temp2)

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