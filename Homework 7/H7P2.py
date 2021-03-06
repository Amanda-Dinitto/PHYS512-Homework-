import numpy as np 
from matplotlib import pyplot as plt
import scipy
from scipy import signal
plt.ion()
def green_theorem(n):
    pot = np.zeros([n,n])
    x = np.linspace(-1*(n//2), n//2, n)
    xx, yy = np.meshgrid(x,x)
    r = np.sqrt(xx**2 + yy**2)
    r[n//2,n//2] = 1.0e-8 ##to avoid log crashing at log(0)
    pot = np.log(r)/2*np.pi    ##equation for potential
    pot[0,0] = 4*pot[1,0] - pot[2,0] - pot[1,1] - pot[1,-1]  ##using neighbors of V[1,0] to solve for singularity potential 
    C = 1/pot[0,0]
    pot = pot*C  # update with scaling factor so pot[0,0] = 1.0
    print('Potential at [5,0] is', pot[5,0])
    return pot 

n = 99
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
"""Plot BC to check it makes a square"""
plt.imshow(pot)
plt.title('Potential Boundary Conditions')
#plt.savefig('H7P2B_BC.png')
b = new_b(pot, mask)
rho = conjugate_gradient(b, V, mask, iter=30). ##Calculate rho via conjugate gradient 
plt.plot(rho[n//3,:]) #plot only one side of box 
plt.title('Charge Density on One Side of Box')
#plt.savefig('H7P2B.png')

"""Part C: Now calculate the potential everywhere using rho and green's"""

Potential = scipy.signal.convolve2d(V, rho)
print(Potential[n//5:n//3, n//5:n//3])
plt.imshow(Potential)
plt.colorbar()
plt.title('Potential')
#plt.savefig('H7P2C.png')
plt.show()



