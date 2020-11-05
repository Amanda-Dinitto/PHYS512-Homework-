#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Oct 30 17:32:03 2020

@author: amanda
"""


import numpy as np
from matplotlib import pyplot as plt
import h5py
import glob
from scipy import signal

## Using Jon's code for reading the ligo data
def read_template(filename):
    dataFile=h5py.File(filename,'r')
    template=dataFile['template']
    th=template[0]
    tl=template[1]
    return th,tl

def read_file(filename):
    dataFile=h5py.File(filename,'r')
    dqInfo = dataFile['quality']['simple']
    qmask=dqInfo['DQmask'][...]
    meta=dataFile['meta']
    #gpsStart=meta['GPSstart'].value
    gpsStart=meta['GPSstart'][()]
    #print meta.keys()
    #utc=meta['UTCstart'].value
    utc=meta['UTCstart'][()]
    #duration=meta['Duration'].value
    duration=meta['Duration'][()]
    #strain=dataFile['strain']['Strain'].value
    strain=dataFile['strain']['Strain'][()]
    dt=(1.0*duration)/len(strain)
    dataFile.close()
    return strain,dt,utc

"""
Problem 1 Homework 5
Using ligo data to search for GW
"""


## Strain data for both Livingston and Hanford detectors
fname_H = 'H-H1_LOSC_4_V2-1128678884-32.hdf5'
fname_L = 'L-L1_LOSC_4_V2-1128678884-32.hdf5'
#print('reading file ',fname_H, 'and', fname_L)
strain_H, dt_H, utc_H = read_file(fname_H)
strain_L, dt_L, utc_L = read_file(fname_L)

## Signal 
template_name='LVT151012_4_template.hdf5'
th,tl=read_template(template_name)

"""
Problem A
smoothing and windowing 
"""
##Create window function
##Hanford
window_H = signal.tukey(len(strain_H)) ##This is a tappered cosine function 
strain_window_H = strain_H*window_H
sH_FT = np.fft.rfft(strain_window_H)
##Livingston
window_L = signal.tukey(len(strain_L))
strain_window_L = strain_L*window_L
sL_FT = np.fft.rfft(strain_window_L)

##Smoothing/whitening using a guassian filter and boxcar filter 
#Hanford
estimate_H = np.abs(sH_FT)**2
eFT_H = np.fft.rfft(estimate_H)
npix = 10
vec = np.zeros(len(estimate_H))
"""
gaussian filter
#x = np.arange(0,1,0.1)
#vec[:npix] = np.exp(-0.5*x**2/(0.5)**2)
"""
##Boxcar filter 
vec[:npix] = 1
vec[-npix+1:]=1
vec = vec/np.sum(vec)
vec_FT_H = np.fft.rfft(vec)
N_H = np.fft.irfft(vec_FT_H*eFT_H, len(estimate_H))
NH = np.maximum(estimate_H, N_H)
#plt.loglog(np.abs(NH))
#plt.savefig('Noise for H-H1_LOSC_4_V2-1128678884-32png')

#Livingston
estimate_L = (np.abs(sL_FT)**2)
eFT_L = np.fft.rfft(estimate_L)
"""
gaussian filter
#x = np.arange(0,1,0.1)   
#vec[:npix] = np.exp(-0.5*x**2/(0.2**2))
"""
##Boxcar filter 
vec[:npix] = 1
vec[-npix+1:]=1
vec = vec/np.sum(vec)
vec_FT_L = np.fft.rfft(vec)
N_L = np.fft.irfft(vec_FT_L*eFT_L, len(estimate_L))
NL = np.maximum(estimate_L, N_L)
#plt.loglog(np.abs(NL)) 
#plt.savefig('Noise for L-L1_LOSC_4_V2-1128678884-32.png')

"""
Problem B
Matched filter processing the events
"""
#Hanford
A_H = window_H*th  ##have to window the data as well 
AFT_H = np.fft.rfft(A_H)
matched_filter_H = np.fft.irfft(np.conj(AFT_H)*(sH_FT/NH))
noise_H = np.std(matched_filter_H)
s2n_H = np.max(np.abs(matched_filter_H))/noise_H ##SNR for matched filter 
print(s2n_H)
#plt.plot(matched_filter_H)
#plt.savefig('MF_H-H1_LOSC_4_V2-1128678884-32.png')

#Livingston
A_L = window_L*tl ##windowing data 
AFT_L = np.fft.rfft(A_L)
matched_filter_L = np.fft.irfft(np.conj(AFT_L)*(sL_FT/NL))
noise_L = np.std(matched_filter_L)
s2n_L = np.max(np.abs(matched_filter_L))/noise_L ##SNR for matched filter 
print(s2n_L)
#plt.plot(matched_filter_L)
#plt.savefig('MF_L-L1_LOSC_4_V2-1128678884-32.png')

"""
Problem C 
Combined SNR
"""
##using a quadrature sum to get combined SNR for both Hanford and Livinston 
combined = np.sqrt((s2n_H)**2 + (s2n_L)**2) 
print('combine SNR is', combined) 

"""
Problem d
SNR analytic for Noise model  
"""
n_H = np.max(np.abs(matched_filter_H))/(np.sqrt(np.mean((np.fft.irfft(AFT_H/np.sqrt(NH)))**2)))
print(n_H)
n_L = np.max(np.abs(matched_filter_L))/(np.sqrt(np.mean((np.fft.irfft(AFT_L/np.sqrt(NL)))**2)))
print(n_L)

"""
Problem e
Frequency 
"""
F_H = np.real((AFT_H/(np.sqrt(NH)))**2)
freq = np.fft.fftfreq(F_H.size, dt_H)
plt.plot(freq, np.cumsum(F_H))

