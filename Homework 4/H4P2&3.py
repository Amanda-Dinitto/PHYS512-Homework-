#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Oct 19 20:32:14 2020

@author: amanda
"""


import numpy as np
from matplotlib import pyplot as plt

def correlation(f, g):
    F = np.fft.fft(f)
    conj = np.fft.fft(g)
    G = np.conjugate(conj)
    H = np.fft.ifft(F*G)
    return H 

x = np.arange(0,10,0.1)

def gaussian(x, dx): 
    func = np.exp(-0.5*(x+dx)**2/(0.75**2))
    return func

func1 = gaussian(x, 0) #not shifted 
func2 = gaussian(x, -10) #shifted 
##Problem 2: correlation of gaussian with itself 
f1 = func1/func1.sum()
g1 = func1/func1.sum()
H1 = np.real(correlation(f1,g1))
##Problem 3: Correlation of shifted gaussian with itself 
f2 = func2/func2.sum()
g2 = func2/func2.sum()
H2 = np.real(correlation(f2, g2))
##Plot to compare change in correlation between shifted and non-shifted 
plt.plot(x, H1) 
plt.plot(x, H2)
plt.title('Shifted correlation vs regular correlation')
#plt.title('Correlation of F and G')
#plt.savefig('H4P23.png')
plt.show() 
