#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Sep 30 19:34:59 2020

@author: amanda
"""


import numpy as np
from matplotlib import pyplot as plt 

##Load data in for each variable 
dish = np.loadtxt("./dish_zenith.txt")
x = dish[:,0]
y = dish[:,1]
z = dish[:,2]

xx = np.square(x)
yy = np.square(y)

##paraboloid in cylindrical coordinates 
z_0 = 0
rr = (xx + yy)
r = np.sqrt(rr)
a = (z-z_0)/rr
zz_true = z_0 + a*rr # zz_true is the same thing as z just checking the calculation for a was good 
f_true = rr/(4*zz_true)
#print(np.average(f_true))

def mat_A(r, order):
    A = np.zeros([len(r), order+1])
    A[:,0] = 1
    for i in range(order):
        A[:,i+1]=r*A[:,i]
    return A

##Set up least squares fit 
##2nd degree polynomial so order of 2
A = mat_A(r, 2)
N = np.eye(len(r))
Ninv = np.linalg.inv(N)
lhs = A.transpose()@(Ninv@A)
rhs = A.transpose()@(Ninv@zz_true) 
m = np.linalg.inv(lhs)@rhs
pred_z = A@m


##Estimate Noise and Uncertainty 
rms = np.std(zz_true - pred_z)
Noise = rms**2
Ninva = N/Noise ##noise matrix using error from solution
print("the noise is", Noise)
a_pred = pred_z/rr
uncertainty_a_2 = np.sqrt(np.diag(np.linalg.inv(A.transpose()@Ninva@A)))
print("uncertainty in a is", uncertainty_a_2[2]) ##since we only have 2nd degree term 



##Focal Length and Error Bars 
##Not sure about the error bar in this section need a new A but not sure how to do it 
f = rr/(4*pred_z)
print ('the focal length is', np.average(f))
rms_f = np.std(f_true - f) ##Noise for focal length specifically 
Ninv2 = N/(rms_f**2) 
E = A.transpose()@(Ninv2@A)
error = np.sqrt(np.diag(np.linalg.inv(E)))
print ('the error bar is', error[2]) ##only 2nd degree term 

plt.plot(r,zz_true, '.', color = 'k', label = 'data')
plt.plot(r,pred_z, '.', label = 'fit')
plt.legend()
plt.title('Linear Least Squares Fit')
plt.savefig('H3P1.jpg')
plt.show()


