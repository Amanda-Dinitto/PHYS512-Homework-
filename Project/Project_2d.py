import numpy as np
from matplotlib import pyplot as plt 


def greens_fn(n): 
    dx = np.arange(n)
    dx[n//2:]=dx[n//2:]-n
    pot=np.zeros([n,n])
    xmat,ymat = np.meshgrid(dx,dx)
    dr=np.sqrt(xmat**2+ymat**2)
    dr[0,0]=1 #dial something in so we don't get errors
    pot=np.log(dr)/(2*np.pi)
    pot = 1/(dr*4*np.pi)
    #pot=pot-pot[n//2,n//2]  #set it so the potential at the edge goes to zero 
    #pot[0,0]=pot[0,1]-0.25 #we know the Laplacian in 2D picks up rho/4 at the zero point
    flip = n//2
    pot[]
    return pot
   
def density_grid(x, y, n):
    points = (np.real(x),np.real(y))
    #print(points)
    grid_min = 0
    grid_max = n
    n_grid = n
    H, edges_x ,edges_y = np.histogram2d(x,y, bins = n_grid, range=([grid_min, grid_max], [grid_min, grid_max]))
    return H
    

def get_potential(x, y, n):
   greens = greens_fn(n)
   rho  = density_grid(x, y, n)
   G = np.fft.fftn(greens)
   r = np.fft.fftn(rho)
   potential = np.fft.ifftn(G*r)
   #print(np.real(potential))
   return potential

def get_force(xx, yy, m, n, soft = 0.0):
    pot = get_potential(x, y, n)
    fx = np.zeros([len(x)])
    fy = np.zeros([len(x)])
    dx, dy = np.gradient(pot) 
    for i in range(0,len(x)): 
        x_floor = int(np.round(np.real(x[i])))
        y_floor = int(np.round(np.real(y[i])))
        ax = np.real(dx[(x_floor), (y_floor)])
        ay = np.real(dy[(x_floor), (y_floor)])
        fx[i] = -(ax)*m
        fy[i] = -(ay)*m
    #print('force', fx, fy)
    return fx, fy, pot  

def take_leapfrog_step(x, y, vx, vy, dt, m, n):
    """take half step""" 
    xx = x + 0.5*((vx)*dt)
    yy = y + 0.5*((vy)*dt)
    fx, fy, pot = get_force(xx, yy, m, n, soft = 0.0)
    for i in range(0,len(x)):
        vvx = vx[i] + 0.5*(dt*fx[i])
        vvy = vy[i] + 0.5*(dt*fy[i])
        """update all values"""
        x[i] = x[i] + (dt*vvx)
        y[i] = y[i] + (dt*vvy)
        vx = vx + (dt*fx[i])
        vy = vy + (dt*fy[i])
   # print(x, y)
    return x, y, vx, vy, pot

m=1
dt = 0.005
n = 20
#x = np.asarray([4.0,5.0])
#y = np.asarray([4.0,4.0])
x = np.random.randn(2)
y = np.random.randn(2)


#vx = np.ones([len(x)])*0.25
#vy = np.ones([len(y)])*-0.25
#vx = np.asarray([0.5, -0.5])
#vy = np.asarray([0.5, -0.5])
vx = x*0.0
vy = y*0
#plt.ion()
for iter in range(100):
    x,y,vx,vz,pot = take_leapfrog_step(x,y,vx,vy,dt,m,n)
    ke = 0.5*m*(np.sum(vx**2 + vy**2))
    #conservation = ke + np.real(np.average(pot))
    #print(conservation)
    plt.plot(x,y,'.')
    plt.xlim(-5,5)
    plt.ylim(-5,5)
    plt.pause(0.1)
plt.show()
plt.clf() 

    
    