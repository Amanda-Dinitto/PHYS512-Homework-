#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Sep 14 15:29:19 2020

@author: amanda
"""
##Problem 2

import numpy as np
from matplotlib import pyplot as plt 

##ignoring the last column dV/dT since it is not necessary 
list = np.loadtxt("./lakeshore2.txt", usecols=range(0,2))
#reduced version of lakeshore file used to display how accurate a fit since 
#values below temp = 100 very close together 
t = list[:,0]
temp = np.flip(t)
v =list[:,1]
volt = np.flip(v)



V=np.linspace(volt[1],volt[-2],144)
T_interp=np.zeros(len(V))
for i in range (len(V)):
    ind=np.max(np.where(V[i]>=volt)[0])
    V_good=volt[ind-1:ind+3]
    T_good=temp[ind-1:ind+3]
    pars=np.polyfit(V_good, T_good,3)
    predicted=np.polyval(pars,V[i])
    T_interp[i]=predicted

plt.plot(V,T_interp )
plt.plot(volt,temp, '.')
plt.xlabel('Voltage')
plt.ylabel('Temperature')
plt.savefig("H1P2_Plot_Short.png")
plt.show()



