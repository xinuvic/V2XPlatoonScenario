# -*- coding: utf-8 -*-
"""
Created on Thu Sep  2 11:09:36 2021

@author: CSUST
"""

def CountConsecutiveNumber(Alist,number,timespot,SimulationTime,BeaconRate,RC,StartTime,maximalTime):

    CountSucceed=0
    CountFail=0
    collision=0
    s=0
    for t in timespot:
        t=t-StartTime
        for j in range(0,len(Alist)):
            s+=1
            if Alist[j][t]==0:
                CountSucceed+=1
            else:
                collision+=1
            # if RC set for last resource selection is no less than 20, regard it as collision
                if t<=SimulationTime-BeaconRate:
                    if sum(Alist[j][t:int(t+maximalTime)])>=maximalTime: 
                        CountFail+=1
                    else:
                        CountSucceed+=1
    return CountSucceed,CountFail,collision,s


def CounterConsecutiveNumber(Alist):
    accum_0_list=[]
    accum_0_list_del0=[]
    accumulate_0=0
    accumulate_1=sum(Alist)
    for i in range(0,len(Alist)-1):
        if sum(Alist[i:i+2])==0:
            accumulate_0+=1          
        else:
            accum_0_list.append(accumulate_0)
            accumulate_0=0
        if 1<=i<len(Alist)-1 and Alist[i-1]!=0 and Alist[i]==0 and Alist[i+1]!=0:
            accum_0_list.append(100)
    for i in accum_0_list:
        if i!=0 and i!=100:
            accum_0_list_del0.append(i+1)
        if i==100:
            accum_0_list_del0.append(1)
    if len(accum_0_list_del0)==0:accum_0_list_del0=[0]
    return accum_0_list_del0,accumulate_1

def Delay_list(Alist,TransmitInterval):#输入Alist是一串连续时隙碰撞与否的列表，如果为0则为碰撞，如果为数值，则为接入时延，接入时延与选择时隙有关
    accum_0_list=[]
    accum_1_list=[]
    if Alist[0]!=0:
        accum_1_list.append(Alist[0])
    for i in range(1,len(Alist)-1):
        if Alist[i-1]!=0:#前一个不是0，前一个成功
            if Alist[i]==0:#从当前为0的时刻开始，找到0截止的时刻，统计本段持续为0的次数accu_0
                accu_0=1
                for s in range(i,len(Alist)):
                    if Alist[s]==0:
                        accu_0+=1
                    else:
                        break
                accum_0_list.append((accu_0-1)*TransmitInterval+Alist[s])
            else:
                accum_1_list.append(Alist[i])
    if Alist[-2]!=0 and Alist[-1]!=0:
        accum_1_list.append(Alist[-1])
    return accum_0_list+accum_1_list
                
                
