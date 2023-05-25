#!/usr/bin/env python3
# -*- coding: utf-8 -*-

# -*- coding: utf-8 -*-
"""
Created on Thu Sep  2 11:17:46 2021

@author: CSUST
"""
import numpy as np

def RSSIratePercent(i,AverageRSSI,ResourceLastList,SubchannelNum,available_res_ratio):
    num_need = int(available_res_ratio * SubchannelNum)
    temp=[]
    w=[]
    Inf = 1000
    p = AverageRSSI
    q = p[:]
    s = min(p)
    #print('s:',s)
    for kkk in range(0,SubchannelNum):  
        w.append(q.index(s))
        q[q.index(s)]=Inf
        if s not in q:
            break
    if len(w)>num_need:
        temp = np.random.choice(w,size = num_need,replace = False)
    else:
        for sss in range(0,num_need):
            temp.append(p.index(min(p)))
            p[p.index(min(p))]=Inf # exclude the recorded maximum number
    return temp  

def RSSIratePercent_front(i,AverageRSSI,ResourceLastList,SubchannelNum,available_res_ratio,delay):
    SubchannelNum_front = int(delay*2)
    num_need = int(available_res_ratio * SubchannelNum_front)
    temp=[]
    w=[]
    Inf = 1000
    p = AverageRSSI[:SubchannelNum_front]
    q = p[:]
    s = min(p)
    #print('s:',s)
    for kkk in range(0,SubchannelNum_front):  
        w.append(q.index(s))
        q[q.index(s)]=Inf
        if s not in q:
            break
    if len(w)>num_need:
        temp = np.random.choice(w,size = num_need,replace = False)
    else:
        for sss in range(0,num_need):
            temp.append(p.index(min(p)))
            p[p.index(min(p))]=Inf # exclude the recorded maximum number
    return temp