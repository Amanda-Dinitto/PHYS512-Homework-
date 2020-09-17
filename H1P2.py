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
list = np.loadtxt("./lakeshore.txt", usecols=range(0,2))
#reduced version of lakeshore file used to display how accurate a fit since 
#values below temp = 100 very close together 
temp = list[:,0]
volt =list[:,1]
#print (temp)
#print (volt)


V=np.linspace(temp[1],temp[-2],144)
T_interp=np.zeros(len(V))
for i in range (len(V)):
    ind=np.max(np.where(V[i]>=temp)[0])
    T_good=volt[ind-1:ind+3]
    V_good=temp[ind-1:ind+3]
    pars=np.polyfit(V_good, T_good,3)
    predicted=np.polyval(pars,V[i])
    T_interp[i]=predicted

plt.plot(V,T_interp )
#plt.plot(volt,temp, '.')
plt.xlabel('')
#plt.savefig("H1P2_Plot.png")
plt.show()


##error calculation
##simple subtraction between interpolated val and array val
estimate = (T_interp)-(temp)
rough = np.std(T_interp-temp)
print ("the standard deviation is between interpolation and actual values is", rough)

