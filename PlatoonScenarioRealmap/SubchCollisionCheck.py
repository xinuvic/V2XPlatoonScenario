# -*- coding: utf-8 -*-
"""
Created on Thu Sep  2 11:15:25 2021

@author: CSUST
"""
def SubchCollisionCheck(i,j,R):
    if R[i] == R[j] :
        Same = 1
    else:
        Same = 0
    return Same