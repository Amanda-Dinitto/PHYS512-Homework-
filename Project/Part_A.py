
import numpy as np 
from matplotlib import pyplot as plt 

def greens2(n):
    dx1 = np.arange(n//2+1) ##we want array from -n/2 to n/2 but no negative 
    dx2 = np.flip(dx1)[:-1] ###flip array but exclude last value to avoid 2 zeros 
    dx = np.concatenate((dx2,dx1)) ##add the arrays together
    pot = np.zeros([n,n])
    soft = 0.1
    for i in range(n):
        for j in range(n):
            dr=np.sqrt(dx[i]**2+dx[j]**2) #create grid of distances from r 
            if dr < soft:
                dr = soft
            pot[i,j]=1/(dr) 
    return pot

def get_potential(x, n, G):
    GFT = np.fft.fft2(G)
    rho, edge_x, edge_y = np.histogram2d(x[:,0], x[:,1], bins = n, range=([0,n], [0,n]))
    rhoFT = np.fft.fft2(rho)
    m = GFT*rhoFT
    pot = np.abs(np.fft.fftshift(np.fft.ifft2(m)))
    return pot

def take_leapfrog_step(x, v, dt, m, n, num_par, G):
    potential = get_potential(x,n, G)
    a = np.gradient(potential) 
    for iter in range(20):
        vv = v.copy()
        for i in range(2):
            vv[:,i] = v[:,i] + ((a[i][x[:,0].astype(int),x[:,1].astype(int)])/(m[:])*(dt/2)) ##x(i=0) and y(i=1) velocity 
        x = x +(vv*dt)
        potential = get_potential(x,n,G)
        a = np.gradient(potential)
        for i in range(2):
            vv[:,i] = v[:,i] + ((a[i][x[:,0].astype(int),x[:,1].astype(int)])/(m[:])*(dt/2)) 
        plt.plot(x[:,0], x[:,1], '.')
        plt.title('1 particle at rest for iteration' + str(iter))
        plt.savefig('1_particle_iter'+str(iter)+'.png')
        ke = 0.5*m[i]*(np.sum(v[:,0]**2 + v[:,1]**2))
        u = np.sum(potential)
        E_tot = ke + u
        print(E_tot)
    return x, v
   

"""
One particle position scaling np.random so 
particle not on edge of plot
"""
m=np.ones(1)
dt = 0.1
n = 60
x = n*np.random.rand(1,2)
num_par = 1
v = x*0
g = greens2(n)
x,v = take_leapfrog_step(x, v,dt,m,n, num_par, g)
plt.show()
plt.clf() 

    
