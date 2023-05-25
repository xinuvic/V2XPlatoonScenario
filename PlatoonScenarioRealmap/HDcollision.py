# -*- coding: utf-8 -*-
"""
Created on Tue Feb  1 15:25:11 2022

@author: CSUST
"""
def HDcollision(r1,r2):
    if abs(r1-r2)==1 and min(r1,r2)%2==0:
        return 1
    else: return 0