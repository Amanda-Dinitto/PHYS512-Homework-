#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Sep 23 20:32:38 2020

@author: amanda
"""


import numpy as np
from scipy import integrate
from matplotlib import pyplot as plt
import time
def U238(x,y,half_life=[4,2e-11,4e-15,7e-5,2e-5,5e-7,3e-12,1e-15,1e-14,1e-14,1e-20,7e-9,1e-9,1e-10]):
    dydx=np.zeros(len(half_life)+1)
    dydx[0]=-y[0]/half_life[0]
    dydx[1]=y[0]/half_life[0]-y[1]/half_life[1]
    dydx[2]=y[1]/half_life[1]-y[2]/half_life[2]
    dydx[3]=y[2]/half_life[2]-y[3]/half_life[3]
    dydx[4]=y[3]/half_life[3]-y[4]/half_life[4]
    dydx[5]=y[4]/half_life[4]-y[5]/half_life[5]
    dydx[6]=y[5]/half_life[5]-y[6]/half_life[6]
    dydx[7]=y[6]/half_life[6]-y[7]/half_life[7]
    dydx[8]=y[7]/half_life[7]-y[8]/half_life[8]
    dydx[9]=y[8]/half_life[8]-y[9]/half_life[9]
    dydx[10]=y[9]/half_life[9]-y[10]/half_life[10]
    dydx[11]=y[10]/half_life[10]-y[11]/half_life[11]
    dydx[12]=y[11]/half_life[11]-y[12]/half_life[12]
    dydx[13]=y[12]/half_life[12]-y[13]/half_life[13]
    dydx[14]=y[13]/half_life[13]
    return dydx


y0=np.asarray([1,0,0,0,0,0,0,0,0,0,0,0,0,0,0]) 
x0=0
x1=1

t1=time.time()
implicit=integrate.solve_ivp(U238,[x0,x1],y0,method='Radau')
t2=time.time()
print('took ',implicit.nfev,' evaluations and ',t2-t1,' seconds to solve implicitly')
print('final value is',implicit.y[0,-1])
plt.plot(implicit.t, implicit.y[1,:])
plt.savefig('H2P3A.jpg')
plt.show()


    
