# -*- coding: utf-8 -*-
"""
Created on Tue Feb  1 22:01:28 2022

@author: CSUST
"""
# i counted from 0 to PlatoonLen-1
def Neigh4IndexSet(i,FirstIndex,PlatoonLen):
    inplatoon_index=i-FirstIndex
    neighset=[]
    if inplatoon_index>=PlatoonLen:
        print('wrong input, i should be no more than PlatoonLen-1')
    elif inplatoon_index==0:
        neighset.append(i+1)
        neighset.append(i+2)
    elif inplatoon_index==1:
        neighset.append(i-1)
        neighset.append(i+1)
        neighset.append(i+2)
    elif inplatoon_index==PlatoonLen-1:
        neighset.append(i-1)
        neighset.append(i-2)
    elif inplatoon_index==PlatoonLen-2:
        neighset.append(i+1)
        neighset.append(i-1)
        neighset.append(i-2)
    else:
        neighset.append(i-1)
        neighset.append(i-2)        
        neighset.append(i+1)
        neighset.append(i+2)
    return neighset


def NeighIndexSet(i,FirstIndex,PlatoonLen,num):
    neighset=list(range(i-num,i+num+1))
    neighset_new=[]
    #print(neighset)
    for item in neighset:
        if FirstIndex<=item<FirstIndex+PlatoonLen and item!=i:
            neighset_new.append(item)
            #print(neighset_new)
    return neighset_new
