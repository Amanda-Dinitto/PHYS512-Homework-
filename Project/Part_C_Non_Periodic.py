import numpy as np 
from matplotlib import pyplot as plt 

def greens2(n):
    dx1 = np.arange(n//2+1) ##we want array from -n/2 to n/2 but no negative 
    dx2 = np.flip(dx1)[:-1] ###flip array but exclude last value to avoid 2 zeros 
    dx = np.concatenate((dx2,dx1)) ##add the arrays 
    pot = np.zeros([n,n])
    soft = 0.1
    for i in range(n):
        for j in range(n):
            dr=np.sqrt(dx[i]**2+dx[j]**2)
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
    pot = get_potential(x,n, G)
    potential = pot[250:1250, 250:1250]
    #mask = np.zeros([len(pot), len(pot)], dtype=bool)
    #mask[250:1250, 250:1250] = True
    #mask[0:250, 0:250] = False
    #mask[1250:1500, 1250:1500] = False 
    #potential = pot[mask]
    a = np.gradient(potential) 
    for iter in range(3000):
        vv = v.copy()
        for i in range(2):
            vv[:,i] = v[:,i] + ((a[i][x[:,0].astype(int),x[:,1].astype(int)])/(m[:])*(dt/2)) ##x velocity 
        x = (x +(vv*dt))
        pot = get_potential(x,n,G)
        potential = pot[250:1250, 250:1250]
        #mask = np.zeros([len(pot), len(pot)], dtype=bool)
        #mask[250:1250, 250:1250] = True
        #mask[0:250, 0:250] = False
        #mask[1250:1500, 1250:1500] = False 
        #potential = pot[mask]
        a = np.gradient(potential)
        for i in range(2):
           v[:,i] = vv[:,i] + ((a[i][x[:,0].astype(int),x[:,1].astype(int)])/(m[:])*(dt/2)) ##x velocity 
        plt.imshow(potential)
        plt.title('N_body at End Non Periodic')
        ke = 0.5*m[i]*(np.sum(v[:,0]**2 + v[:,1]**2))
        u = np.sum(potential)
        E_tot = ke + u
        print(E_tot)
    return x, v


"""
Hundreds of thousand particles now. Velocity will start
at rest and evolve
"""
num_par = 100000
dt = 0.01
n = 1000
m=np.ones(num_par)
x = n*np.random.rand(num_par,2)
v = x*0 
N = 1500
g = greens2(N)
x,v = take_leapfrog_step(x, v,dt,m,N, num_par, g)
plt.show()
plt.clf() 

    


