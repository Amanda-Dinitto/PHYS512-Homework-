#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Sep 24 20:39:25 2020

@author: amanda
"""


import numpy as np
from scipy import integrate
def lorentz(x):
    return 1/(1+x**2)

def x3(x): 
    return x**3

def integrate_step(fun,x1,x2,y1, y2, tol):
    print('integrating from ',x1,' to ',x2)
    x=np.linspace(x1,x2,5)
    y=fun(x)
    area1=(x2-x1)*(y1+4*y[2]+y2)/6
    area2=(x2-x1)*( y1+4*y[1]+2*y[2]+4*y[3]+y2)/12
    myerr=np.abs(area1-area2)
    if myerr<tol:
        return area2
    else:
        xm=0.5*(x1+x2)
        ym=fun(xm)
        a1=integrate_step(fun,x1,xm,y1, ym,tol/2)
        a2=integrate_step(fun,xm,x2,ym,y2,tol/2)
        return a1+a2
    
#ans_x3= integrate_step(x3, 0, 2,0, 8, 0.001)
#ans_x3_true = integrate.quad(x3, 0, 2)
#error_x3 = ans_x3 - ans_x3_true[0]

ans = integrate_step(lorentz, -1, 1,0.5, 0.5, 0.001)
ans_true = integrate.quad(lorentz, -1, 1)
error = ans - ans_true[0]

print(ans, ans_true, error)
#print( ans_x3, ans_x3_true, error_x3)