#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Sep 15 16:24:44 2020

@author: amanda
"""


import numpy as np
from matplotlib import pyplot as plt
from scipy import interpolate
from scipy.interpolate import interp1d
#Cos function
val_1 = np.linspace(-np.pi/2, np.pi/2, 6)
func_1 = np.cos(val_1)

#Lorentzian Fucntion
val_2= np.linspace(-1,  1, 4)
func_2 = 1/(1+(val_2)**2)


##Ploynomial Interpolation
#cos function
x = np.linspace(val_1[0],val_1[-1],50)
f1 = interp1d(val_1, func_1, kind='cubic')
#lorentzian function
z = np.linspace(val_2[0],val_2[-1], 50)
f2 = interp1d(val_2, func_2, kind='cubic')


##Cubic Spline 
#cos fucntion
spln_1 = interpolate.splrep(val_1, func_1)
interp_1=interpolate.splev(x,spln_1)
#lorentzian function
spln_2 = interpolate.splrep(val_2, func_2)
interp_2 = interpolate.splev(z,spln_2)

##Rational Interpolation
#Cos Function
def rat_eval_1(p,q,val_1):
    top=0
    for i in range(len(p)):
        top=top+p[i]*val_1**i
    bot=1
    for i in range(len(q)):
        bot=bot+q[i]*val_1**(i+1)
    return top/bot

def rat_fit_1(val_1,func_1,n,m):
    assert(len(val_1)==n+m-1)
    assert(len(func_1)==len(val_1))
    mat=np.zeros([n+m-1,n+m-1])
    for i in range(n):
        mat[:,i]=val_1**i
    for i in range(1,m):
        mat[:,i-1+n]=-func_1*val_1**i
    pars=np.dot(np.linalg.inv(mat),func_1)
    p=pars[:n]
    q=pars[n:]
    return p,q
n=3
m=4
p,q =rat_fit_1(val_1, func_1, n,m)
xx = np.linspace(val_1[0],val_1[-1],50)
func_true_1= np.cos(xx)
predicted_1= rat_eval_1(p,q,xx)

#Lorentzian Function
def rat_eval_2(p,q,val_2):
    top=0
    for i in range(len(p)):
        top=top+p[i]*val_2**i
    bot=1
    for i in range(len(q)):
        bot=bot+q[i]*val_2**(i+1)
    return top/bot

def rat_fit_2(val_2,func_2,n,m):
    assert(len(val_2)==n+m-1)
    assert(len(func_2)==len(val_2))
    mat=np.zeros([n+m-1,n+m-1])
    for i in range(n):
        mat[:,i]=val_2**i
    for i in range(1,m):
        mat[:,i-1+n]=-func_2*val_2**i
    pars=np.dot(np.linalg.inv(mat),func_2)
    p=pars[:n]
    q=pars[n:]
    return p,q
n=2
m=3
p,q =rat_fit_2(val_2, func_2, n,m)
zz = np.linspace(5*val_2[0],5*val_2[-1],50)
func_true_2= 1/(1+(zz)**2)
predicted_2= rat_eval_2(p,q,zz)
error = predicted_2 - func_true_2
print ("Standard deviation for Lorentz Rational fit is", np.std(error)*100)


#Graph for Cos comparison
plt.plot(val_1, func_1, 'o', label='Cos(x) Points')
plt.plot(x, f1(x), '-', label='Poly Interp')
plt.plot(x, interp_1, '-', color='k', label='Spline')
plt.plot(xx, predicted_1,'-', label='Rational')
plt.legend(loc='upper right')
plt.title("Cos(x) Comparison")
#plt.savefig('H1P3_Cos.jpg')
plt.show()



#Graph comparison for Lorentzian Function
plt.plot(val_2, func_2, 'o', label='Lorentz Points')
plt.plot(z, f2(z), label='Poly')
plt.plot(z, interp_2,'--', color='k', label='Spline')
plt.plot(zz, predicted_2,'*', label='Rational')
plt.legend(loc = 'upper right')
plt.title("Lorentzian Comparison")
#plt.savefig('H1P3_Lorentz_Bad.jpg')
plt.show()




