#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Jun 14 15:31:27 2019

@author: wuzhiqiang
提取同一时长下相同电流点来的电压变化特征，供内短路分析用
"""

def filter_data1(data, upper_err, lower_err, border='min'):
    """
    给定一组数据和容忍误差范围，所有在边界内的数值全部改写为边界值，靠近哪边改为哪边
    默认最小边界以最小的数为基准
    如：data：[17.9，18.1, 15.9, 20.1, 23.1,18.1],upper_err=0.4, lower_err=1.2
    return:[17.5, 17.5, 15.9, 19.1, 23.9, 19.1]
    """
    if len(data) <= 1:
        return data
    if border == 'min':
        interval = upper_err + lower_err
        first_border = min(data)
        end_border = max(data) + interval
    elif border == 'max':
        interval = - upper_err - lower_err
        first_border = max(data)
        end_border = min(data) + interval
        
    border_seq = []
    border = first_border
    while border < end_border:
        border_seq.append(round(border, 1))
        border += interval
    return border_seq

def filter_data(data, bias):
    """
    给定一组数据和容忍误差范围,根据bias生成边界，落入边界内的所有数据均认为相等，并标记为下边界
    从0开始向两边展开边界
    如：data：[17.9，18.1, 15.9, 20.1, 23.1,18.1],upper_err=0.4, lower_err=1.2
    return:[17.5, 17.5, 15.9, 19.1, 23.9, 19.1]
    """
    if len(data) <= 1:
        return data
    temp1 = min(data)
    temp2 = max(data)
    if (temp2 - temp1) <= bias:
        data1 = temp1
        return data, data1
    else:
        
    
a = [17.9, 18.1, 15.9, 20.1, 23.1, 18.1]
b = filter_data1(a, 0.4, 1.2, 'min')
print(b)
    