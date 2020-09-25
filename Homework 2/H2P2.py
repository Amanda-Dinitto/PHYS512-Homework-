#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Sep 22 17:07:05 2020

@author: amanda
"""
import numpy as np
from matplotlib import pyplot as plt
def log2(x):
    return np.log2(x)

## real y value 
x = np.linspace(0.5,1,51)
y_true = log2(x)

## coeffs is commented out since the code is calculating the error right now 
def chebyshev(func,a, b, ord):
    x = np.linspace(0.5, 1, ord+1)
    u = (2*x -a-b)/(b-a)
    y = func(x)
    mat = np.zeros([ord+1, ord+1])
    mat[:,0]=1
    mat[:,1]=u
    
    for i in range(1, ord):
        mat[:,i+1]=2*u*mat[:,i]-mat[:,i-1]
    mat_inverse = np.linalg.inv(mat)
    coeffs = ((np.dot(mat_inverse, y)))
    return mat
    #return coeffs

#coeffs = chebyshev(log2,0.5, 1, 50)
mat = chebyshev(log2, 0.5,1,50)

##predicted y value for chebyshev
N = 50
lhs = np.dot(mat.transpose(), mat)
rhs = np.dot(mat.transpose(), y_true)
fit = np.dot(np.linalg.inv(lhs), rhs)
mat = mat[:,1::2]
fit = fit[1::2]
pred = np.dot(mat[:,:N],fit[:N])


def polyfit(func, ord):
    x = np.linspace(0.5,1, ord+1)
    y = func(x)
    poly_fit = np.polynomial.legendre.legfit(x, y, ord)
    return poly_fit

poly_fit = polyfit(log2, 50)


##graphing and errors


rms_chebyshev = np.sqrt(np.mean(pred - y_true)**2)
max_chebyshev = np.max(np.abs(pred - y_true))
print("the rms error is for chebyshev fit is",rms_chebyshev,"and the max error is", max_chebyshev)
rms_poly_fit = np.sqrt(np.mean(poly_fit - y_true)**2)
max_poly_fit = np.max(np.abs(poly_fit - y_true))
print("the rms error for polynomial legendre fit is", rms_poly_fit,"and the max error is", max_poly_fit)
plt.plot(x,pred-y_true)
plt.plot(x, poly_fit-y_true)
plt.savefig("H2P2.jpg")
plt.show()
