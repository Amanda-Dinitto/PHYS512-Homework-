import numpy as np 
from matplotlib import pyplot as plt 
from mpl_toolkits import mplot3d
from matplotlib import cm 





def greens_fn(n, soft = 0.01 ): 
    dx = np.arange(n)
    dx[n//2:]=dx[n//2:]-n
    pot=np.zeros([n,n,n])
    xmat,ymat, zmat=np.meshgrid(dx,dx,dx)
    dr=np.sqrt(xmat**2+ymat**2 + zmat**2)
    #print(dr)
    dr[0,0,0]=1 #dial something in so we don't get errors
    #if dr < soft:
    #    dr = soft 
    pot=1/(4*np.pi*dr)
    pot_soft= 1/(4*np.pi*soft)
    if pot > pot_soft: 
        pot = pot_soft
    #pot=pot-pot[n//2,n//2]  #set it so the potential at the edge goes to zero is this padding??
    #assert 1==0
    return pot
    
def density_grid(x, y, z, n):
    points = (x,y,z)
    grid_min = -n/2
    grid_max = n/2
    n_grid = n
    H, edges = np.histogramdd(points, bins = n_grid, range=((grid_min, grid_max), (grid_min, grid_max), (grid_min, grid_max))) 
    return H
    

def get_potential(x, y, z, n, soft = 0.01):
   greens = greens_fn(n, soft = 0.01)
   rho = density_grid(x, y, z, n)
   G = np.fft.fftn(greens)
   r = np.fft.fftn(rho)
   potential = np.fft.ifftn(G*r)
   return potential, greens

def get_force(x, y, z, m, n, soft = 0.1):
    potential, greens = get_potential(x, y, z, n)
    for i in range(len(x)):
        
        pot = potential - np.roll(greens, [x,y,z], axis = (0,1,2))
        #assert 1==0
        dx = np.gradient(pot, axis=0) 
        dy = np.gradient(pot, axis=1)
        dz = np.gradient(pot, axis=2)
        ax = dx[x,y,z]
        ay = dy[x,y,z]
        az = dz[x,y,z]
        a_max = 1/soft**2
        #print(a_max)
        """softening"""
        if ax > a_max: 
            ax = a_max
        if ay > a_max: 
            ay = a_max
        if az > a_max :
            az = a_max 
        fx = (ax)*m
        fy = (ay)*m
        fz = (az)*m
    return fx, fy, fz, pot   

def take_leapfrog_step(x, y, z, vx, vy, vz, dt, m, n):
    """take half step"""
    xx = x + 0.5*vx*dt
    yy = y + 0.5*vy*dt
    zz = z + 0.5*vz*dt
    fx, fy, fz, pot = get_force(xx, yy, zz, m, n)
    vvx = vx + 0.5*dt*fx
    vvy = vy + 0.5*dt*fy
    vvz = vz + 0.5*dt*fz
    """update all values"""
    x = x + dt*vvx
    y = y + dt*vvy
    z = z + dt*vvz
    vx = vx + dt*fx
    vy = vy + dt*fy
    vz = vz + dt*fz
    return x,y,z, vx, vy, vz, pot


"""
#Part A: Single particle at rest
"""
n = 3 ##this is size of grid NOT number of particles 
m = 1
##the following points describe the single particle position 
x = np.array([0])
y = np.array([0])
z = np.array([0])
vx = 0.0 #velocity is zero 
vy = 0.0
vz = 0.0
soft = 0.01
dt = soft**1.5 ##dt given by v_max/a_max = soft**1.5
#pot = 0
for iter in range(10):
    x,y,z, vx, vy, vz, pot = take_leapfrog_step(x,y,z,vx,vy,vz,dt, m, n)
    KE = 0.5*np.sum(m*(vx**2 + vy**2 + vz**2)) 
##now plt particle position over time 
    #print(np.real(x), np.real(y), np.real(z))
    plt.ion()
    fig = plt.figure()
    ax = plt.axes(projection='3d')
    ax.scatter3D(np.real(x),np.real(y),np.real(z))
    ax.set_xlim(-1e-17,1e-17)
    ax.set_ylim(-1e-17,1e-17)
    ax.set_zlim(-1e-17,1e-17)
    #plt.savefig('single point iteration'+ str(iter)+'.png')
    plt.show()
    
"""
Part B: 2 Particles in Circular Orbit
##work in progress still

n = 10
m = 1.0
x = np.array([0,1])
y = np.array([0,1])
z = np.array([0,1])
v???

soft = 0.01
dt = soft**1.5*0.05 ##dt given by v_max/a_max = soft**1.5
for iter in range(10):
    x,y,z, vx, vy, vz, pot = take_leapfrog_step(x,y,z,vx,vy,vz,dt, m, n)
    KE = 0.5*np.sum(m*(vx**2 + vy**2 + vz**2)) 
    #print(np.real((pot-KE)/2))
##now plt particle position over time 
    plt.ion()
    fig = plt.figure()
    ax = plt.axes(projection='3d')
    ax.scatter3D(np.real(x),np.real(y),np.real(z))
    ##set limits
    #plt.savefig('single point iteration'+ str(iter)+'.png')
    plt.show()


Part C: Many Particles with Periodic and Non-Periodic Boundary Conditions
Due to the wrap-around nature of FFT's the code as is is periodic. Non-Periodic 
can be done by eliminating said wrap-around nature when taking the convolution.
This is done by padding the functions we are convolving with zeros. 
"""


