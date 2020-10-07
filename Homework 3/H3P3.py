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



def get_spectrum(pars,lmax=2000):
    #print('pars are ',pars)
    H0=pars[0]
    ombh2=pars[1]
    omch2=pars[2]
    tau=pars[3]
    As=pars[4]
    ns=pars[5]
    pars=camb.CAMBparams()
    pars.set_cosmology(H0=H0,ombh2=ombh2,omch2=omch2,mnu=0.06,omk=0,tau=tau)
    pars.InitPower.set_params(As=As,ns=ns,r=0)
    pars.set_for_lmax(lmax,lens_potential_accuracy=0)
    results=camb.get_results(pars)
    powers=results.get_cmb_power_spectra(pars,CMB_unit='muK')
    cmb=powers['total']
    tt=cmb[:,0]    #you could return the full power spectrum here if you wanted to do say EE
    return tt


##parameters and the differencing step 
pars=np.asarray([65,0.02,0.1,0.05,2e-9,0.96])
#pars_fixed = np.asarray([])
delta_pars = pars*(np.asarray([0.5,0.5,0.5,0.0,0.5,0.5])) ##half step with tau held constant 
#delta_pars = pars*(np.asarray([0.0,0.0,0.0,0.5,0.0,0.0])) ##half step with everything other than tau held constant 

def get_derivs(pars):
    pars_d = pars.copy()
    x = get_spectrum(pars_d)
    cmb = x[2:1201] ##truncated
    derivs = np.zeros([len(cmb),len(pars)]) ##the matrix 
    x_u = np.array(get_spectrum(pars_d + delta_pars))
    cmb_up = x_u[2:1201] ##truncated 
    x_d = np.array(get_spectrum(pars_d - delta_pars))
    cmb_down = x_d[2:1201] ##truncated 
    derivs[:,0] = (cmb_up - cmb_down)/2*(delta_pars[0])#H0
    derivs[:,1] = (cmb_up - cmb_down)/2*(delta_pars[1]) #ombh2
    derivs[:,2] = (cmb_up - cmb_down)/2*(delta_pars[2]) #omch2
    derivs[:,3] = (cmb_up - cmb_down)/2*(delta_pars[3]) #tau
    derivs[:,4] = (cmb_up - cmb_down)/2*(delta_pars[4]) #As
    derivs[:,5] = (cmb_up - cmb_down)/2*(delta_pars[5]) #ns  
    return derivs 

#derivs = get_derivs(pars)
#plt.plot(derivs[:,5])
#plt.title('Derivative plot for ns')
#plt.savefig('H3P3_ns.jpg')
#plt.show()



data=np.loadtxt('wmap_tt_spectrum_9yr_v5.txt')
wmap = data[:,1]
error = data[:,2]
N = np.eye(len(error))
Ninv = np.linalg.inv(N)

pars_g = pars.copy() ##first guess for optimization
#pars_f = pars_fixed.copy() ##array with best fit terms 
for iter in range(2):
    x = get_spectrum(pars_g)
    cmb = x[2:1201] #cut out values after 1200 term since wmap is short
    derivs = get_derivs(pars_g)
    residuals = wmap - cmb
    rhs = derivs.transpose()@Ninv@residuals
    lhs = derivs.transpose()@Ninv@derivs
    step = (np.linalg.inv(lhs))@rhs
    #print(step)
    pars_step = pars_g + step
    #print(pars_step, 'for iteration', iter)

pars_error=np.sqrt(np.diag(np.linalg.inv(lhs)))
print('the final parameters are', pars_step[0], 'for H0', pars_step[1], 'for ombh^2', pars_step[2], 'for omch^2', pars_step[4], 'for As and', pars_step[5], 'for ns')
print('the errors per term are', pars_error)
print('the final tau is', pars_step[3])
  
