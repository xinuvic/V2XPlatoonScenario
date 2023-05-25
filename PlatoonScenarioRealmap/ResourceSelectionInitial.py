# -*- coding: utf-8 -*-
"""
Created on Thu Sep  2 11:18:48 2021

@author: CSUST
"""
import random
def ResourceSelectionInitial(NumVehicle,SubchannelNum,aviod_collision_ini):
    ResourceSelectionall = [[]]*NumVehicle
    selected=[]
    if aviod_collision_ini==True:
        for i in range(0,NumVehicle):
            #ResourceSelectionall[i] = random.randint(0,SubchannelNum-1)
            if i<=SubchannelNum-1:
                ResourceSelectionall[i] = random.randint(0,SubchannelNum-1)
                while ResourceSelectionall[i] in selected:
                    ResourceSelectionall[i] = random.randint(0,SubchannelNum-1)
                    #print('ResourceSelectionall[i]',i,ResourceSelectionall[i])
                selected.append(ResourceSelectionall[i])
            else:
                ResourceSelectionall[i] = ResourceSelectionall[i-SubchannelNum]
    else:
        for i in range(0,NumVehicle):
            #ResourceSelectionall[i] = random.randint(0,SubchannelNum-1)
            if i<=SubchannelNum-1:
                ResourceSelectionall[i] = random.randint(0,SubchannelNum-1)
                selected.append(ResourceSelectionall[i])
            else:
                ResourceSelectionall[i] = ResourceSelectionall[i-SubchannelNum]
    return ResourceSelectionall