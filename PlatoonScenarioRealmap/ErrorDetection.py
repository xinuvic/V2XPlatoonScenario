# -*- coding: utf-8 -*-
"""
Created on Tue Feb  1 15:14:30 2022

@author: CSUST
"""
#from CalculateSINR import CalculateSINR
from CalculateSINR_fading import CalculateSINR_fading,CalculateSINR
from HDcollision import HDcollision
from Neigh4IndexSet import NeighIndexSet


def ErrorDetection(i,FirstIndex,PlatoonLen,ResourceSelectionall,NumVehicle,VehicleLocation,power,sinr_th,num,fading,scale_param_omega):
    CollisionDetectedfromPM = 0
    CollisionConfirm = 0
    false_alarming = 0
    # neighset is the set of transmitters in the neighbourhood
    neighset = NeighIndexSet(i,FirstIndex,PlatoonLen,num)
    Loss_from_collision=0
    #print('for',i,'neighset is',neighset)
    for j in neighset:
        #if CalculateSINR(i,j,ResourceSelectionall,NumVehicle,VehicleLocation,power)<sinr_th:
        if CalculateSINR_fading(i,j,ResourceSelectionall,NumVehicle,VehicleLocation,power,fading,scale_param_omega)<sinr_th:
            #print('detect packet loss from',i,'to',j)
            if CalculateSINR_fading(j,i,ResourceSelectionall,NumVehicle,VehicleLocation,power,fading,scale_param_omega)>=sinr_th and HDcollision(ResourceSelectionall[i],ResourceSelectionall[j])!=1:
                CollisionDetectedfromPM = CollisionDetectedfromPM + 1
                #print('feedback is sent from',j,'to',i)
        if CalculateSINR(i,j,ResourceSelectionall,NumVehicle,VehicleLocation,power)<sinr_th:
            Loss_from_collision += 1        
    if CollisionDetectedfromPM >= 1:
        CollisionConfirm = 1
        print('collision detected +1')
    else:
        CollisionConfirm = 0
    if CollisionConfirm==1 and Loss_from_collision==0:
        false_alarming = 1
        print('false alarm counted +1')

    return CollisionConfirm,false_alarming
