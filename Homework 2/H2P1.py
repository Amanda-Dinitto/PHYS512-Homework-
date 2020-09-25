#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Sep 22 16:35:22 2020

@author: amanda
"""
import numpy as np
from scipy import integrate
def lorentz(x):
    return 1/(1+x**2)

def x4(x): 
    return x**3

def integrate_step(fun,x1,x2,tol):
    print('integrating from ',x1,' to ',x2)
    x=np.linspace(x1,x2,5)
    y=fun(x)
    area1=(x2-x1)*(y[0]+4*y[2]+y[4])/6
    area2=(x2-x1)*( y[0]+4*y[1]+2*y[2]+4*y[3]+y[4])/12
    myerr=np.abs(area1-area2)
    if myerr<tol:
        return area2
    else:
        xm=0.5*(x1+x2)
        a1=integrate_step(fun,x1,xm,tol/2)
        a2=integrate_step(fun,xm,x2,tol/2)
        return a1+a2
    
ans_cos = integrate_step(x4, 0, 2, 0.001)
ans_cos_true = integrate.quad(x4, 0, 2)
error_cos = ans_cos - ans_cos_true[0]

ans = integrate_step(lorentz, -1, 1, 0.001)
ans_true = integrate.quad(lorentz, -1, 1)
error = ans - ans_true[0]

print(ans, ans_true, error)
#print( ans_cos, ans_cos_true, error_cos)