#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Oct 12 17:24:35 2020

@author: amanda
"""


import numpy as np
import camb
from matplotlib import pyplot as plt
import time


data=np.loadtxt('wmap_tt_spectrum_9yr_v5.txt')

##this is our function
def get_spectrum(pars,lmax=2000):
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

##our chisq from problem 2 
def chisq(data, pars):
    wmap = data[:,1]
    error = data[:,2]
    x = get_spectrum(pars)
    cmb = x[2:1201] #cut out values after 1200 term since wmap is short
    ##Solve for chi^2 now (x-m)^2/error^2
    chisq = np.sum((wmap - cmb)**2/error**2)
    return chisq

##numerical derivative for Newton's method 
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
    return derivs 

##MCMC 
def mcmc(pars, data, pars_step, chisq_func, cov_mat, nstep):
    chain = np.zeros([nstep, len(pars)])
    chi_vector = np.zeros(nstep)
    chi_orig = chisq_func(data, pars)
    r = 0.7*(np.linalg.cholesky(cov_mat))
    for i in range(nstep):
        d = np.random.randn(r.shape[0])
        while d[5] < 0:
            #print('still neg')
            d = np.random.randn(r.shape[0])
            if d[5] > 0: 
                break
        d = d 
        #print(d)
        pars_test = pars + np.dot(r,d)
        chi_test = chisq_func(data, pars_test)
        chi_gaussian = (pars_test[5] - 0.0544)**2/0.0073**2 ##gaussian likelihood for tau 
        accept = np.exp(-0.5*(chi_test - chi_orig)*chi_gaussian) ##full likelihood now 
        if np.random.rand(1)<accept:
            pars = pars_test
            chi_orig = chi_test
        chain[i,:] = pars
        chi_vector[i] = chi_orig
    return chain, chi_vector 
    
    
##parameters and the differencing step 
pars_orig=np.asarray([65,0.02,0.1,2e-9,0.96, 0.05])
pars = pars_orig.copy()
delta_pars = pars*np.asarray([0.01,0.01,0.01,0.01,0.01,0.01])  
data = np.loadtxt('wmap_tt_spectrum_9yr_v5.txt')
wmap = data[:,1]
error = data[:,2]
N = np.eye(len(error))
Ninv = np.linalg.inv(N)

 
##chain sampling from our Newton method
x = get_spectrum(pars)
cmb = x[2:1201] 
derivs = get_derivs(get_spectrum, cmb, pars, delta_pars)
old_lhs = derivs.transpose()@Ninv@derivs
cov_mat = np.linalg.inv(old_lhs) 
Q,R = np.linalg.qr(derivs)
lhs = Q.transpose()@Ninv@Q@R
lhs_inv = np.linalg.inv(lhs)
pars_step = np.sqrt(np.diag(np.abs(lhs_inv))) ## use curvature matrix to make estimate for step size 
chain, chi_vector = mcmc(pars, data, pars_step, chisq, cov_mat, nstep=10000)
error = np.std(chain, axis=0)
pars_better = np.mean(chain, axis=0)
parameters = np.mean(chain, axis = 0)
print("the parameter estimates using mcmc are", parameters)
print( "error for first chains is",error)








