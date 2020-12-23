import numpy as np 
from matplotlib import pyplot as plt 

def greens2(n):
    dx1 = np.arange(n//2+1) ##we want array from -n/2 to n/2 but no negative 
    dx2 = np.flip(dx1)[:-1] ###flip array but exclude last value to avoid 2 zeros 
    dx = np.concatenate((dx2,dx1)) ##add the arrays 
    pot = np.zeros([n,n])
    soft = 0.01
    for i in range(n):
        for j in range(n):
            dr=np.sqrt(dx[i]**2+dx[j]**2)
            if dr < soft:
                dr = soft
            pot[i,j]=1/(dr) 
    return pot

def get_potential(x, n, G):
    GFT = np.fft.fft2(G) ##scaled greens to make potential stronger
    rho, edge_x, edge_y = np.histogram2d(x[:,0], x[:,1], bins = n, range=([0,n], [0,n]))
    rhoFT = np.fft.fft2(rho)
    m = GFT*rhoFT
    pot = np.abs(np.fft.fftshift(np.fft.ifft2(m)))
    return pot

def take_leapfrog_step(x, v, dt, m, n, num_par, G):
    potential = get_potential(x,n, G)
    a = np.gradient(potential) 
    for iter in range(800):
        vv = v.copy()
        for i in range(num_par):
            vv[i, 0] = v[i,0] + ((a[0][x[i,0].astype(int),x[i,1].astype(int)])/(m[i])*(dt/2)) ##x velocity 
            vv[i, 1] = v[i,1] + ((a[1][x[i,0].astype(int),x[i,1].astype(int)])/(m[i])*(dt/2)) ##y velocity 
        x = x +(vv*dt)
        potential = get_potential(x,n,G)
        a = np.gradient(potential)
        for i in range(num_par):
            v[i, 0] = vv[i,0] + ((a[0][x[i,0].astype(int),x[i,1].astype(int)])/(m[i])*(dt/2)) ##x velocity 
            v[i, 1] = vv[i,1] + ((a[1][x[i,0].astype(int),x[i,1].astype(int)])/(m[i])*(dt/2)) ##y velocity 
        #plt.plot(x[:,0], x[:,1], '.')
        #plt.title('2 Particles in circular orbit')
        ke = 0.5*m[i]*(np.sum(v[:,0]**2 + v[:,1]**2))
        u = np.sum(potential)
        E_tot = ke + u
        print(E_tot)
    return x, v
   
m=np.ones(2)
dt = 0.01
n = 100
num_par = 2
x = np.zeros([2,2])
##place 2 particles 10 units apart towards middle of grid
x[:,0] = n/2
x[0,1] = 29.5
x[1,1] = 30.5
v = np.zeros([2,2])
v[0,0] = 0.55
v[1,0] = -0.55
#pot= get_potential(x,n)
#plt.imshow(pot)
g = greens2(n)
plt.title('Greens Function Potential')
plt.savefig('Greens_Pot.png')
plt.imshow(g)

x,v = take_leapfrog_step(x, v,dt,m,n, num_par, g)



plt.show()
plt.clf() 

    