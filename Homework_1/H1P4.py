#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Sep 16 14:59:32 2020

@author: amanda
"""


import numpy as np
from matplotlib import pyplot as plt
from scipy import integrate
R = 1.0 ##radius of sphere
##Constants from equation are ignored 
##Scipy Quad
##this works but for some reason cant get a line to go through points 
## function should die right after surface but increases first????
for z_in in [0, 0.2, 0.4, 0.6, 0.8, 0.99]:
    def Gauss_In(x):
        num_in = np.sin(x)*((z_in) - R*np.cos(x))
        denom_in = R**2 + (z_in)**2 -2*R*(z_in)*np.cos(x)
        return num_in/(denom_in)**(3/2)
    Func_true_in = integrate.quad(Gauss_In, 0, np.pi)
    print(Func_true_in[0], 'for', z_in)
    plt.plot(z_in, Func_true_in[0], 'o') 
   
z_surface = 1.0
def Gauss_Surface(x):
    num_surface = np.sin(x)*((z_surface) - R*np.cos(x))
    denom_surface = R**2 + (z_surface)**2 -2*R*(z_surface)*np.cos(x)
    return num_surface/(denom_surface)**(3/2)
Func_true_surface = integrate.quad(Gauss_Surface, 0, np.pi)
print(Func_true_surface[0], "for", z_surface)

plt.plot(z_surface, Func_true_surface[0], '*')
  

for z_out in [1.1, 1.2, 1.3, 1.4, 1.6, 1.8, 2.0]:
    def Gauss_Out(x):
        num_out = np.sin(x)*((z_out) - R*np.cos(x))
        denom_out = R**2 + (z_out)**2 -2*R*(z_out)*np.cos(x)
        return num_out/(denom_out)**(3/2)
    Func_true_out = integrate.quad(Gauss_Out, 0, np.pi)
    print(Func_true_out[0], 'for', z_out)
    plt.plot(z_out, Func_true_out[0], 'o')


##Built Integration Routine
def integrate_step(func, xL, xU, tol):
    x = np.linspace(xL, xU, 5)
    y = func(x)
    A1 = (xU-xL)*(y[0]+4*y[2]+y[4])/6
    A2 = (xU-xL)*(y[0]+4*y[1]+2*y[2]+4*y[3]+y[4])/12
    error = np.abs(A1-A2)
    if error<tol:
        return A2
    else:
        x_mid = (xL+xU)/2
        f_left = integrate_step(func, xL, x_mid, tol/2)
        f_right = integrate_step(func, x_mid, xU, tol/2)
        f_total = f_left + f_right
        return f_total

##At z=1.0 code doesnt run since there is a discontinuity when z = R 
for z_in in [0, 0.2, 0.4, 0.6, 0.8, 0.99]:
    Integral_In = integrate_step(Gauss_In, 0, np.pi, 0.0001)
    print (Integral_In, "for", z_in)
for z_out in [1.1, 1.2, 1.4, 1.6, 1.8, 2.0]:
    Integral_Out = integrate_step(Gauss_Out, 0, np.pi, 0.0001)
    print(Integral_Out, "for", z_out)
    
## My plot kept giving me an error so unfortunately I had to 
## just write out my points by hand so that I could do the polyfit,
## if you know what I did wrong so that I couldn't use my variables
## please let me know!! Thanks!!
z1 = np.array([0, 0.2, 0.4, 0.6, 0.8, 0.99])
Int1 = np.array([3.2061178228696494e-17, 5.945604400481308e-07 , -1.4605163247694009e-06, -1.7443664747096577e-06,
                -1.0159703347500937e-06, -1.0485203303933766e-06])
C = np.polyfit(z1, Int1, 1)
poly1 = np.poly1d(C)
x = np.linspace(z1[0], z1[-1])
y = poly1(x)
plt.plot(z1, Int1,'.', x, y, color='k')

z2 = np.array([1.1, 1.2, 1.4, 1.6, 1.8, 2.0])
Int2 = np.array([1.652894955590231, 1.3888896970246434, 1.0204086264204177, 0.7812518809282596, 0.6172847329823478, 0.5000009334814289])
D = np.polyfit(z2, Int2, 3)
poly2 = np.poly1d(D)
x2 = np.linspace(z2[0], z2[-1])
y2=poly2(x2)
plt.plot(z2, Int2, '.', x2, y2, color='k')
plt.xlabel("Distance from centre (z)")
plt.ylabel("Electric field")
plt.savefig("H1P4.jpg") 
plt.show()    





