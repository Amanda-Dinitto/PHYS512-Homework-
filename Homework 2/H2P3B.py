#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Sep 23 23:25:38 2020

@author: amanda
"""


import numpy as np
from scipy import integrate
import time
from matplotlib import pyplot as plt

def U238(x,y,half_life=[4]):
    dydx=np.zeros(len(half_life)+1)
    dydx[0]=-y[0]/half_life[0]
    dydx[1]=y[0]/half_life[0]
    return dydx

def U234(x,y,half_life=[2.45e-4]):
    dydx=np.zeros(len(half_life)+1)
    dydx[0]=-y[0]/half_life[0]
    dydx[1]=y[0]/half_life[0]
    return dydx

y0=np.asarray([1,0]) 
x0=0
x1=1
t1=time.time()
implicit1=integrate.solve_ivp(U238,[x0,x1],y0,method='Radau')
t2=time.time()

t1=time.time()
implicit2=integrate.solve_ivp(U234,[x0,x1],y0,method='Radau')
t2=time.time()
print('took ',implicit1.nfev,' evaluations and ',t2-t1,' seconds to solve implicitly')
print('final value is',implicit1.y[0,-1])


print('took ',implicit2.nfev,' evaluations and ',t2-t1,' seconds to solve implicitly')
print('final value is',implicit2.y[0,-1])
###plotting???
plt.plot(implicit1.t, implicit1.y[0,:]/implicit1.y[1,:],'--')
plt.plot(implicit2.t, implicit2.y[0,:]/implicit2.y[1,:],)
plt.xlabel('time')
plt.xlim(-0.01,0.6)
plt.savefig('H2P3B_Zoom_In.jpg')
plt.show()
    
