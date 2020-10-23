#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Oct 20 22:39:36 2020

@author: amanda
"""


import numpy as np
from matplotlib import pyplot as plt
"""
Problem 5C: compare the DFT equation to the FFT from numpy 
"""
##Setting up values 
j = np.complex(0,1)
N=100
k= np.fft.fftfreq(N,1/N) ##Non integer value for k
a = (2*np.pi/3)

##DFT estimate
num1 =  1 - (np.exp(-j*((2*np.pi*k) - a*N)))
denom1 = 1 - (np.exp(-j*(((2*np.pi*k)/N) - a)))
num2 =  1 - (np.exp(-j*((2*np.pi*k) + a*N)))
denom2 = 1 - (np.exp(-j*(((2*np.pi*k)/N) + a)))
frac1 = (num1)/(denom1)
frac2 = (num2)/(denom2)
dft = ((1/(2*j))*(frac1 - frac2))


##Using FFT now 
x=np.arange(N)
T = 3
f = np.sin((2*np.pi*x/T))
fft = np.fft.fft(f)
print('this is the original FFT', np.real(fft))
##difference between numpy and estimate 
error = np.mean(np.abs(dft - fft)) 
print('difference is', error)

##Plotting

plt.plot(np.real(dft),'.')
plt.plot(np.real(fft))
plt.title('Fourier transform with DFT eqn and FFT')
plt.savefig('H4P5c.png')
plt.show()

"""
Problem 5D and E: Adding a window to the function f from 
our fft by adding the window we should notice less 
leakage at the edges
"""
window = 0.5 - 0.5*np.cos(2*np.pi*x/N)
win = np.fft.fft(window) ##FT of the window 
print('this is the FT of win', np.real(win))
f_win = f*window
fft_win = np.fft.fft(f_win) ##FFT with window 


##Plotting
plt.clf()
plt.plot(np.real(fft), '.')
plt.plot(np.real(fft_win), '*')
plt.title('Spectral Leakage with vs Without Window')
plt.savefig('H4P5e.png')
plt.show()