#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Sep 11 18:18:56 2020

@author: amanda
"""
###Problem 1 Part B

import numpy as np 

## for f = exp(x)
x=1
#real derivative of f
real=np.exp(x)
print ('The actual value for exp(x) is', real)
#delta value
d = 10**-4 
#separate the equation into 4 parts
fp=np.exp(x+d)
fm=np.exp(x-d)
fp2=np.exp(x+(2*d))
fm2=np.exp(x-(2*d))

f1=(fp-fm)/(2*d)
f2=(fp2-fm2)/(2*d)
#estimated derivative
estimate=(f2-f1)
print ('The estimate value is', estimate)
print ('The difference between estimated and actual value', np.abs(estimate - real))
    


## same thing but for f =exp(0.01x) now 

x0=1
x=0.01*x0
#real derivative of f
real=np.exp(0.01*x)
print ("The actual value for exp(0.01x) is", real)
#delta value
d = 10**-4 
#separate the equation into 4 parts
fp=np.exp(x+d)
fm=np.exp(x-d)
fp2=np.exp(x+(2*d))
fm2=np.exp(x-(2*d))

f1=(fp-fm)/(2*d)
f2=(fp2-fm2)/(2*d)
#estimated derivative
estimate=(f2-f1)
print ("The estimate value is", estimate)
print ('The difference between estimatede and actual value', np.abs(estimate - real))