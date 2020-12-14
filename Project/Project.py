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

def density_grid(particles, cells):
    grid=np.zeros(cells)
    cell_length=1.0/cells
    left_index_f = 0.0
    cell_position= 0.0
    for i in range(particles.shape[0]):
        left_index=particles[i]
        (left_index_f,cell_position)=divmod((particles[i]),cell_length)
        cell_position = cell_position/cell_length
        left_index=int(left_index_f)
        grid[left_index % cells]+=1-cell_position
        grid[(left_index + 1) % cells]+=cell_position
    return np.array([grid])

def get_potential(x, y, z, n, soft = 0.01):
   greens = greens_fn(x, y, z, soft = 0.01)
   rho = density_grid(x, n)
   G = np.fft.fftn(greens)
   r = np.fft.fftn(rho)
   potential = np.fft.ifftn(G*r)
   return potential 

def get_force(x, y, z, m, n, soft = 0.01):
    pot = get_potential(x, y, z, n, soft = 0.01)
    ax, ay, az = np.gradient(pot, 3) ##first order deriv in terms of r
    fx = ax*m
    fy = ay*m
    fz = az*m
    return fx, fy, fz, pot  

def take_leapfrog_step(x, y, z, vx, vy, vz, dt, m, n):
    m = m*np.ones(n)  ##change this 
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
    KE = 0.5*np.sum(m*(vx**2 + vy**2 + vz**2)) 
    return x,y,z,pot, KE


"""
#Part A: Single particle at rest
"""
n = 1 
m = 1.0
x = np.random.randn(n)
y = np.random.randn(n)
z = np.random.randn(n)
vx = 0.0*x #velocity is zero 
vy = 0.0*y
vz = 0.0*z
soft = 0.01
dt = soft**1.5 ##dt given by v_max/a_max = soft**-1.5

for i in range(10):
    x,y,z,pot, kinetic = take_leapfrog_step(x,y,z,vx,vy,vz,dt, m, n)
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
