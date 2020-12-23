import numpy as np 
from matplotlib import pyplot as plt 
from mpl_toolkits import mplot3d




def greens_fn(n): 
    dx1 = np.arange(n//2+1) ##we want array from -n/2 to n/2 but no negative 
    dx2 = np.flip(dx1)[:-1] ###flip array but exclude last value to avoid 2 zeros 
    dx = np.concatenate((dx2,dx1)) ##add the arrays 
    pot = np.zeros([n,n,n])
    soft = 0.1
    for i in range(n):
        for j in range(n):
            for k in range(n):
                dr=np.sqrt(dx[i]**2 + dx[j]**2 + dx[k]**2)
                if dr < soft:
                    dr = soft
                pot[i,j]=15/(dr) 
    return pot    
    


def get_potential(x,y,z, g, n):
   points = (x,y,z)
   rho, edges_x, edges_y, edges_z = np.histogramdd(points, bins = n, range=((0,n), (0,n), (0,n))) 
   GFT = np.fft.fftn(g)
   rhoFT = np.fft.fftn(rho)
   potential = np.abs(np.fft.fftshift(np.fft.ifftn(GFT*rhoFT)))
   return potential



def take_leapfrog_step(x,y,z,vx,vy,vz,g,n,num_par):
    potential = get_potential(x,y,z, g, n)
    ax, ay, az = np.gradient(potential)
    for iter in range(100):
        vvx = vx.copy()
        vvy = vy.copy()
        vvz = vz.copy
        for i in range(num_par): 
            vvx[i] = vx[i] + (ax[x[i].astype(int), y[i].astype(int), z[i].astype(int)])*(m[i])*(dt/2)
            vvy[i] = vy[i] + (ay[x[i].astype(int), y[i].astype(int), z[i].astype(int)])*(m[i])*(dt/2)
            vvz[i] = vz[i] + (az[x[i].astype(int), y[i].astype(int), z[i].astype(int)])*(m[i])*(dt/2)
        x = x + (vvx*dt)
        y = y + (vvy*dt)
        z = z + (vvz*dt)
        potential = get_potential(x,y,z,g,n)
        ax, ay, az = np.gradient(potential)
        for i in range(num_par): 
            vx[i] = vvx[i] + (ax[x[i].astype(int), y[i].astype(int), z[i].astype(int)])*(m[i])*(dt/2)
            vy[i] = vvy[i] + (ay[x[i].astype(int), y[i].astype(int), z[i].astype(int)])*(m[i])*(dt/2)
            vz[i] = vvz[i] + (az[x[i].astype(int), y[i].astype(int), z[i].astype(int)])*(m[i])*(dt/2)
        plt.plot(x,y,z)
    return(x,y,z,vx,vy,vz)

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



def k3_mass(num_par): 
    m = np.linspace(0, 1000, 2*num_par)
    noise = 4 + np.random.randn(2*num_par)
    mass = m + noise 
    mass_noise = 1/(mass**3) ##Make k^-3 dependency  ##this behaves like FT therefore IFFT it 
    m = np.abs(np.fft.ifft(mass_noise))
    m = m[num_par//2:-num_par//2]*10**6 ##cutting off edges and scaling so m is bigger than 1 
    return m 

    

"""
#Part B: 2 Particles in Circular Orbit

"""
n = 30
m = np.ones(2) ##mass if one array same length as number of particles 
x = np.asarray([1.,1.])
y = np.asarray([1.,2.])
z = np.asarray([0.,0.]) 
x
vx = np.array([1.,1.])*0.0
vy = np.array([1.,-1.])*0.25
vz = np.array([1.0,-1.0])*0.0
g = greens_fn(n)
dt = 0.1
plt.show()



