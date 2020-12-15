import numpy as np 
from matplotlib import pyplot as plt 


def greens_fn(x, y, z, soft = 0.01): 
    n = len(x)
    for i in range(n):
        for j in range (i,n):
            dx = x[i] - x[j]
            dy = y[i] - y[j]
            dz = z[i] - z[j] 
            rsqr = dx**2 + dy**2 + dz**2
            if rsqr < soft**2: ##add in softening for if r<R 
                rsqr = soft**2 
            r = np.sqrt(rsqr)
            pot = 1/(4*np.pi*r) ## Laplacian in 3D 
    return pot 

def density_grid(x, y, z, n):
    points = (x,y,z)
    #print(points)
    grid_min = -n
    grid_max = n
    n_grid = 2*n
    H, edges = np.histogramdd(points, bins = n_grid, range=((grid_min, grid_max), (grid_min, grid_max), (grid_min, grid_max)))
    edges_x = edges[0]
    edges_y = edges[1]
    edges_z = edges[2] 
    return H 
    

def get_potential(x, y, z, n, soft = 0.01):
   greens = greens_fn(x, y, z, soft = 0.01)
   rho = density_grid(x, y, z, n)
   G = np.fft.fftn(greens)
   r = np.fft.fftn(rho)
   potential = np.fft.ifftn(G*r)
   return potential 

def get_force(x, y, z, m, n, soft = 0.01):
    pot = get_potential(x, y, z, n, soft = 0.01)
    ax = np.gradient(pot, axis=0) 
    ay = np.gradient(pot, axis=1)
    az = np.gradient(pot, axis=2)
    fx = ax*m
    fy = ay*m
    fz = az*m
    return fx, fy, fz, pot  

def take_leapfrog_step(x, y, z, vx, vy, vz, dt, m, n):
    #m = m*np.ones(n)  ##change this 
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
n = 1
m = 1
x = np.array([0])
y = np.array([0])
z = np.array([0])
vx = 0.0*x #velocity is zero 
vy = 0.0*y
vz = 0.0*z
soft = 0.01
dt = soft**1.5 ##dt given by v_max/a_max = soft**-1.5

for iter in range(10):
    x,y,z, vx, vy, vz, pot = take_leapfrog_step(x,y,z,vx,vy,vz,dt, m, n)
    #KE = 0.5*np.sum(m*(vx**2 + vy**2 + vz**2)) 
##now plt particle position over time 
    
"""
Part B: 2 Particles in Circular Orbit
##work in progress still

n = 2
m = 1.0
x = np.random.randn(n)
y = np.random.randn(n)
z = np.random.randn(n)

r = np.sqrt(x**2 + y**2 + z**2) ##radius is this right?
m1 = m2 = 1.0 ##mass of 2 particles 
g = 1.0 #gravitational constant is it necessary? 
v = np.sqrt((g*(m1+m2))/r) ## velocity in circular motion
v = 0.0*v  ??? how to separate the velocity components?
dt = soft**1.5*0.05 ##dt given by v_max/a_max = soft**1.5
"""

