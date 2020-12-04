#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Dec  3 17:40:22 2020

@author: amanda
"""


import numpy as np 
from matplotlib import pyplot as plt
import scipy
from scipy import signal
plt.ion()

def green_theorem(n):
    pot = np.zeros([n,n])
    rho = np.zeros([n,n])
    x = np.arange(n)
    xx, yy = np.meshgrid(x,x)
    r = np.sqrt(xx**2 + yy**2)
    r[0,0] = 1.0e-8    ##Avoid code crashing because of log(0)
    pot = np.log(r)/2*np.pi    ##equation for potential
    pot=pot-pot[n//2,n//2]     ##make potential zero at edges 
    pot[1,0] = (pot[0,0] + pot[2,0] + pot[1,1] + pot[1,-1])/4
    pot[0,0] = 4*pot[1,0] - pot[2,0] - pot[1,1] - pot[1,-1]  ##using neighbors solve for singularity potential 
    C = 1/pot[0,0]
    pot[0,0] = C*(4*pot[1,0] - pot[2,0] - pot[1,1] - pot[1,-1]) # add scaling factor so pot[0,0] = 1.0
    pot[1,0] = C + (pot[0,0] + pot[2,0] + pot[1,1] + pot[1,-1])/4
    #rho = pot - (np.roll(pot,1,axis=0) + np.roll(pot,-1,axis=0) + np.roll(pot,1,axis=1) + np.roll(pot,-1, axis=1))/4  ##equation for rho 
    #rho[0,0] = 1.0
    #pot = pot*rho ##update pot equation 
    pot[5,0] = C + (pot[4,0] + pot[6,0] + pot[5,-1] + pot[5,1])/4   ##Sanity check 
    print('Potential at [5,0] is', pot[5,0])
    return pot 

n = 100
V = green_theorem(n)

"""
For B we are using a conjugate gradient to solve the matrix equation Ax=b where here 
V=convolution(greens, rho) so rho is the value we are solving for. 
"""

def new_Ax(x, mask): 
    x_new = x.copy()
    x[mask]= 0
    avg = np.roll(x_new, 1, axis=0) + np.roll(x_new, -1, axis=0) + np.roll(x_new, 1, axis=1) + np.roll(x_new, -1, axis=1)
    x_new = x_new - avg/4
    return x_new

def new_b(x, mask): 
    x_new = x.copy()
    """make sure everything not defined by boundaries is zero""" 
    not_mask=np.logical_not(mask)
    x[not_mask] = 0 
    avg = np.roll(x_new, 1, axis=0) + np.roll(x_new, -1, axis=0) + np.roll(x_new, 1, axis=1) + np.roll(x_new, -1, axis=1)
    avg[mask] = 0
    avg = avg/4
    return avg

def conjugate_gradient(b, xi, mask, iter=10): 
    Ax = new_Ax(xi, mask)
    rk = b-Ax
    pk = rk.copy()
    x = xi.copy()
    rtr = np.sum(rk*rk)
    for i in range(iter): 
        Apk = new_Ax(pk, mask)
        alpha_k = rtr/(np.sum(pk*Apk))
        x_new = x + alpha_k*pk
        r_new = rk - alpha_k*Apk
        rtr_new = np.sum(r_new*r_new)
        beta_k = rtr_new/rtr
        pk= r_new + beta_k*pk
        """update values"""
        rk = r_new 
        print('on iteration', i, 'residual is', rtr_new) ##Check if residual is shrinking
        rtr = rtr_new 
    return x 

"""
In this case our b is the potential and the mask associated to it 
Potential here is zero everywhere except for the box which is being held at 1.0
Our A value is the greens solution from part 1 
The x value is the rho we are solving for 
"""
mask = np.zeros([n,n], dtype='bool')
pot = np.zeros([n,n])
"""Make Potential zero everwhere except square"""
mask[0,:] = True
mask[-1,:] = True
mask[:,0] = True
mask[:,-1] = True 
mask[n//5:n//3, n//5:n//3] = True 
pot[n//5:n//3, n//5:n//3] = 1.0
b = new_b(pot, mask)
rho = conjugate_gradient(b, V, mask, iter=30)
plt.imshow(pot)
plt.title('Potential Boundary Conditions')
plt.savefig('H7P2B_BC.png')
#plt.plot(rho[n//3,:]) #plot only one side of box 
#plt.title('Charge Density on One Side of Box')
#plt.savefig('H7P2B.png')

"""Part C: Now calculate the potential everywhere using rho and green's"""

VV = V + pot ##Include boundary conditions from green's function potential  
Potential = scipy.signal.convolve2d(VV, rho)
#print(Potential[n//5:n//3, n//5:n//3])
#plt.imshow(Potential)
#plt.colorbar()
#plt.title('Potential')
#plt.savefig('H7P2C.png')
plt.show()



