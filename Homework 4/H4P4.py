#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Oct 20 20:21:24 2020

@author: amanda
"""
"""
Circular matrix becomes issue when taking DFT therefore will attempt
to change cyclical nature by substituting zeros to the array that is 
used for the convolution
"""
import numpy as np
from matplotlib import pyplot as plt

def convolution(f, g):
    F = np.fft.fft(f)
    #print(np.real(F))
    G = np.fft.fft(g)
    #print(np.real(G))
    H = np.fft.ifft(F*G)
    return H 

def gaussian(x): 
    func = np.exp(-0.5*(x)**2/(0.75**2))
    return func
##Setting up both f's 
x1 = np.arange(-5, 5, 0.1)
x1_adj = np.arange(-5,4.8, 0.1)
func1 = gaussian(x1)
func1_adj = gaussian(x1_adj)
#print(func1)
f = np.hstack((func1_adj,0,0)) ## subsituting in zeros for end terms
f2 = np.hstack((func1, 0,0)) ##adding zeros to full array 

##Setting up both g's 
x2 = np.arange(-8, 2, 0.1)
x2_adj = np.arange(-8, 1.8, 0.1)
func2 = gaussian(x2)
func2_adj = gaussian(x2_adj)
g = np.hstack((func2_adj,0,0)) ##substituting zero to end of function values
g2 = np.hstack((func2, 0,0)) ##adding zeros instead 

H_orig = np.real(convolution(func1, func2)) ##no adjusting for wrap around
H_adj = np.real(convolution(f, g)) ##adjusted for wrap around by substitution
H_2 = np.real((convolution(f2, g2))) ##adjusted by adding zeros for test 
plt.plot( H_orig)
plt.plot( H_adj, '.')
plt.plot(H_2, 'k')
plt.title('Wrap Around Adjustment')
plt.savefig('H4P4.png')
plt.show()
