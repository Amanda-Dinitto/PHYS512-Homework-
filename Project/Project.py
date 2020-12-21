import numpy as np 
from matplotlib import pyplot as plt 
from mpl_toolkits import mplot3d
from matplotlib import cm 


def greens_fn(n): 
    dx = np.arange(n)
    dx[n//2:]=dx[n//2:]-n
    pot=np.zeros([n,n,n])
    xmat,ymat, zmat=np.meshgrid(dx,dx,dx)
    dr=np.sqrt(xmat**2+ymat**2 + zmat**2)
    dr[0,0,0]=1 #dial something in so we don't get errors
    pot=1/(4*np.pi*dr)
    #pot=pot-pot[n//2,n//2]  #set it so the potential at the edge goes to zero not included so far in any run
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

n = 4 ##this is size of grid NOT number of particles 
m = 1
##the following points describe the single particle 
x = np.array([1])
y = np.array([1])
z = np.array([1])
vx = 0.0*x #velocity is zero 
vy = 0.0*y
vz = 0.0*z
soft = 0.01
dt = 0.05*soft**1.5 ##dt < v_max/a_max = soft**1.5
for iter in range(20):
    x,y,z, vx, vy, vz = take_leapfrog_step(x,y,z,vx,vy,vz,dt, m, n)
    print(np.real(x), np.real(y), np.real(z))

##now plt particle position over time 
    plt.ion()
    fig = plt.figure()
    ax = plt.axes(projection='3d')
    ax.scatter3D(np.real(x),np.real(y),np.real(z))
    plt.title('Single Particle after iteration'+ str(iter))
    #plt.savefig('single point iteration'+ str(iter)+'.png')
    plt.show()

"""
#Part B: 2 Particles in Circular Orbit
#m1v1 = m2v2 for circular motion (or v1/r1 = v2/r2)
##work in progress still
n = 10
m = 1.0
x = np.array([1,2])
y = np.array([1,2])
z = np.array([1,1]) ##same z plane so they will orbit only in x and y 
#velocities should be equal since mass is equal start at zero 
##vx = np.zeros([2])
#vy = np.zeros([2])
#vz = np.zeros([2])
vx = np.array([1,1])*0.25
vy = np.array([1,1])*0.25
vz = np.array([1,1])*0.25

soft = 0.01
dt = soft**1.5*0.5 ##dt given by v_max/a_max = soft**1.5
plt.ion()
fig = plt.figure()
ax = fig.add_subplot(111,projection='3d')
ax.set_xlim(0,4)
ax.set_ylim(0,4)
ax.set_zlim(0,4)
#plt.pause(0.1)
for i in range(50):
    x,y,z, vx, vy, vz, pot = take_leapfrog_step(x,y,z,vx,vy,vz,dt, m, n)
    KE = 0.5*np.sum(m*(vx**2 + vy**2 + vz**2)) 
##now plt particle position over time 
    ax.scatter3D(np.real(x[0]), np.real(y[0]), np.real(z[0]))
    ax.scatter3D(np.real(x[1]), np.real(y[1]), np.real(z[1]))
    #plt.title('2 Particles after iteration '+ str(i))
    #plt.savefig('2 points circular iteration'+ str(iter)+'.png')
plt.show()



"""
Part C: Many Particles with Periodic and Non-Periodic Boundary Conditions
Due to the wrap-around nature of FFT's the code as is is periodic. Non-Periodic 
can be done by eliminating said wrap-around nature when taking the convolution.
This is done by padding the functions we are convolving with zeros. 
"""
