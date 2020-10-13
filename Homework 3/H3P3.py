#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Oct  5 18:12:20 2020

@author: amanda
"""


import numpy as np
import camb
from matplotlib import pyplot as plt
import time
from scipy import optimize 


def get_spectrum(pars,lmax=2000):
    #print('pars are ',pars)
    H0=pars[0]
    ombh2=pars[1]
    omch2=pars[2]
    As=pars[3]
    ns=pars[4]
    tau=pars[5]
    pars=camb.CAMBparams()
    pars.set_cosmology(H0=H0,ombh2=ombh2,omch2=omch2,mnu=0.06,omk=0,tau=tau)
    pars.InitPower.set_params(As=As,ns=ns,r=0)
    pars.set_for_lmax(lmax,lens_potential_accuracy=0)
    results=camb.get_results(pars)
    powers=results.get_cmb_power_spectra(pars,CMB_unit='muK')
    cmb=powers['total']
    tt=cmb[:,0]    #you could return the full power spectrum here if you wanted to do say EE
    return tt

def get_derivs(fun,x,pars,dpars):
    derivs=np.zeros([len(x),len(pars)])
    for i in range(len(pars)):
        pars[i]=pars[i]+dpars[i]
        f1=fun(pars)
        f_right = f1[2:1201]
        pars[i]=pars[i]-dpars[i]
        f2=fun(pars)
        f_left = f2[2:1201]
        derivs[:,i]=(f_right-f_left)/(2*dpars[i])
        #derivs[:,5] = -(f_right-f_left)/(2*dpars[5]) ##forcing tau to be positive 
    return derivs



##parameters and the differencing step 
pars_orig=np.asarray([65,0.02,0.1, 2e-9,0.96, 0.05]) ## pars without tau for first run
pars_fixed = np.asarray([63.25278713574684, 0.022052868015065476, 0.12185338008847825, 2.0807797203976893e-9, 0.9499219480097826, 0.05]) ## pars with only tau for 2nd run 
pars = pars_orig.copy()
#pars = pars_fixed.copy() ##array with best fit terms 
delta_pars = pars*np.asarray([0.01,0.01,0.01,0.01,0.01,0.01]) 

#Plots
#x = get_spectrum(pars)
#cmb = x[2:1201] 
#derivs = get_derivs(get_spectrum, cmb, pars, delta_pars)
#plt.plot(derivs[:,5])
#plt.title('Derivative plot for tau')
#plt.savefig('H3P3_tau.jpg')
#plt.show()


data=np.loadtxt('wmap_tt_spectrum_9yr_v5.txt')
wmap = data[:,1]
error = data[:,2]
N = np.eye(len(error))
Ninv = np.linalg.inv(N)
#Newton's Method and LM
for iter in range(5):
    x = get_spectrum(pars)
    cmb = x[2:1201] 
    ##(A^TN^-1A) = m(A^TN^_1r)
    derivs = get_derivs(get_spectrum, cmb, pars, delta_pars)
    #derivs = np.delete(d,5,1) ## for tau held constant
    Q,R = np.linalg.qr(derivs) ##QR decomposition
    residuals = wmap - cmb
    rhs = Q.transpose()@Ninv@residuals
    lhs = Q.transpose()@Ninv@Q@R ##curvature matrix 
    l = 0 
    chisq = (wmap - cmb)**2/error**2
    for i in range(iter): 
        chisq[i] = (wmap[i] - cmb[i])**2/error[i]**2
        chisq_dif = chisq[i] - chisq[i-1]
        #print(chisq_dif)
        if chisq_dif > 0:
            l = l + 2
        elif chisq_dif < 0:
            l = l - l/np.sqrt(2)
        else: 
            l = l
    l = l 
    diag = np.diag(np.diag(lhs))
    lhs_adjust = lhs + l*diag
    lhs_inv = np.linalg.inv(lhs_adjust) 
    step = lhs_inv@rhs
    ##adding the columns back in 
    #col = 0 ## for tau held constant 
    #col = np.asarray([0,0,0,0,0]) ## for tau float
    #step = np.delete(step, [0,1,2,3,4],0) ## for tau float 
    #full_mat = np.hstack((step, col)) ##for tau constant 
    #full_mat = np.hstack((col, step)) ## for tau float 
    pars = pars + step
    print(pars, 'for iteration', iter)

   
pars_error=np.sqrt(np.diag(np.abs(lhs_inv))) ##sigma^2
print('the final parameters are', pars[0], 'for H0', pars[1], 'for ombh^2', pars[2], 'for omch^2', pars[3], 'for As and', pars[4], 'for ns')
print('the errors per term are', pars_error)
print('the final tau is', pars[5])
  
