# -*- coding: utf-8 -*-
"""
Created on Thu Sep  2 11:16:48 2021

@author: CSUST
"""
from Distance import Distance

def RSSI(i,ResourceLastList,SubchannelNum,NumVehicle,VehicleLocation,power):
    a=[]
    RSSIDistribution = [0]*SubchannelNum        
    for j in range(0,NumVehicle):
        #print(ResourceLastList[j])
        if ResourceLastList[j]==a:
            continue
        k = ResourceLastList[j]
        if i==j or VehicleLocation[i]==VehicleLocation[j]:
            continue
        RSSIValue = power*Distance(i,j,VehicleLocation)**(-3.68)
        if RSSIDistribution[k] == 0:
            RSSIDistribution[k] = RSSIValue
        else:
            RSSIDistribution[k] += RSSIValue
    return RSSIDistribution    