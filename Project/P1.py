import numpy as np 
from matplotlib import pyplot as plt 
from mpl_toolkits import mplot3d
from matplotlib import cm 


def greens_fn(n): 
    dx = np.arange(n)
    pot=np.zeros([n,n,n])
    xmat,ymat, zmat=np.meshgrid(dx,dx,dx)
    dr=np.sqrt(xmat**2+ymat**2 + zmat**2)
    dr[0,0,0]=1 #Softening in order to avoid errors at zero 
    pot=1/(4*np.pi*dr)
    """Flipping the edges to ensure period BC conditions"""
    pot[n//2:,:n//2] = np.flip(pot[:n//2,:n//2], axis=0)
    pot[:,n//2:] = np.flip(pot[:,:n//2], axis = 1)
    return pot
    
def density_grid(x, y, z, n):
    points = (np.real(x),np.real(y),np.real(z))
    grid_min = 0
    grid_max = n
    n_grid = n
    H, edges = np.histogramdd(points, bins = n_grid, range=((grid_min, grid_max), (grid_min, grid_max), (grid_min, grid_max))) 
    return H
    

def get_potential(x, y, z, n):
   greens = greens_fn(n)
   rho = density_grid(x, y, z, n)
   G = np.fft.fftn(greens)
   r = np.fft.fftn(rho)
   potential = np.fft.ifftn(G*r)
   return potential

def get_force(x, y, z, m, n, soft = 0.1):
    pot = get_potential(x, y, z, n)
    dx, dy, dz = np.gradient(pot) 
    """
    Matrices to fill with fx, fy, and fz particle values 
    for each step, therefore they are only the length of 
    the number of particles each. 
    """
    fx = np.zeros([len(x)])
    fy = np.zeros([len(x)])
    fz = np.zeros([len(x)])
    for i in range(0,len(x)): 
        x_floor = int(np.round(np.real(x[i])))
        y_floor = int(np.round(np.real(y[i])))
        z_floor = int(np.round(np.real(z[i])))
        ax = np.real(dx[(x_floor), (y_floor), (z_floor)])
        ay = np.real(dy[(x_floor), (y_floor), (z_floor)])
        az = np.real(dz[(x_floor), (y_floor), (z_floor)])
        fx[i] = -(ax)*m
        fy[i] = -(ay)*m
        fz[i] = -(az)*m
    print(fx, fy, fz)
    return fx, fy, fz, pot    

def take_leapfrog_step(x, y, z, vx, vy, vz, dt, m, n):
    """take half step"""
    xx = x + 0.5*((vx)*dt)
    yy = y + 0.5*((vy)*dt)
    zz = z + 0.5*((vz)*dt)
    fx, fy, fz, pot = get_force(xx, yy, zz, m, n, soft = 0.01)
    for i in range(0,len(x)):
        vvx = vx[i] + 0.5*(dt*fx[i])
        vvy = vy[i] + 0.5*(dt*fy[i])
        vvz = vz[i] + 0.5*(dt*fz[i])
        """update all values"""
        x[i] = x[i] + (dt*vvx)
        y[i] = y[i] + (dt*vvy)
        z[i] = z[i] + (dt*vvz)
        vx = vx + (dt*fx[i])
        vy = vy + (dt*fy[i])
        vz = vy + (dt*fy[i])
    return x,y,z, vx, vy, vz, pot

"""
#Part A: Single particle at rest
"""
nsize = 4 ##this is size of grid NOT number of particles 
m = 1
"""The following points describe the single particle""" 
x = np.array([1])
y = np.array([1])
z = np.array([1])
"""Particle Placed at Rest"""
vx = 0.0*x 
vy = 0.0*y
vz = 0.0*z
dt = 0.01
for iter in range(50):
    x,y,z, vx, vy, vz = take_leapfrog_step(x,y,z,vx,vy,vz,dt, m, nsize)
    print(np.real(x), np.real(y), np.real(z))

##now plt particle position over time 
    plt.ion()
    fig = plt.figure()
    ax = plt.axes(projection='3d')
    ax.scatter3D(np.real(x),np.real(y),np.real(z))
    plt.title('Single Particle after iteration'+ str(iter))
    #plt.savefig('single point iteration'+ str(iter)+'.png')
    plt.show()

