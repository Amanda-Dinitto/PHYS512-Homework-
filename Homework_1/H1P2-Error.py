#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Sep 14 19:56:06 2020

@author: amanda
"""


import numpy as np
from matplotlib import pyplot as plt
##Same code as before but this time separated the list into odd and even points to calculate error better. 
list = np.loadtxt("./lakeshore2.txt", usecols=range(0,2))
# 
t = list[:,0]
temp = np.flip(t)
v =list[:,1]
volt = np.flip(v)
V = np.array(volt[::2])
T = np.array(temp[::2])


V2=np.linspace(V[1],V[-2],47)
T_interp=np.zeros(len(V2))
for i in range (len(V2)):
    ind=np.max(np.where(V2[i]>=V)[0])
    V_good=V[ind-1:ind+3]
    T_good=T[ind-1:ind+3]
    pars=np.polyfit(V_good, T_good,3)
    predicted=np.polyval(pars,V2[i])
    T_interp[i]=predicted
    
plt.plot(V2, T_interp)
plt.plot(V,T, '.')
plt.xlabel('Voltage')
plt.ylabel('Temperature')
plt.savefig("H1P2_Error_Plot_Short.jpg")
plt.show()
    
#interpolation val for every other value and estimated against 
no_interp = np.array(temp[1::2]) #taking the odd values from the lakeshore txt
estimate = (T_interp - no_interp)
trial = np.std(estimate)
print ("The standard deviation between interpolated point and actual value is", trial)
