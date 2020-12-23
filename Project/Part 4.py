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
    potential = get_potential(x,n, G)
    print(len(potential))
    a = np.gradient(potential) 
    for iter in range(1):
        vv = v.copy()
        for i in range(2):
            vv[:,i] = v[:,i] + ((a[i][x[:,0].astype(int),x[:,1].astype(int)])/(m[:])*(dt/2))
        x = (x +(vv*dt))%1000
        potential = get_potential(x,n,G)
        a = np.gradient(potential)
        for i in range(2):
           v[:,i] = vv[:,i] + ((a[i][x[:,0].astype(int),x[:,1].astype(int)])/(m[:])*(dt/2)) 
        plt.imshow(potential)
        plt.title('N_body at Beginning of Iterations_k^3')
        ke = 0.5*m[i]*(np.sum(v[:,0]**2 + v[:,1]**2))
        u = np.sum(potential)
        E_tot = ke + u
        print(E_tot)
    return x, v

def k3_mass(num_par): 
    m = np.linspace(0, 1000, 2*num_par)
    noise = 4 + np.random.randn(2*num_par)
    mass = m + noise 
    mass_noise = 1/(mass**3) ##Make k^-3 dependency  ##this behaves like FT therefore IFFT it 
    m = np.abs(np.fft.ifft(mass_noise))
    m = m[num_par//2:-num_par//2]*10**6 ##cutting off edges and scaling so m is bigger than 1 
    return m 


"""
Hundreds of thousand particles now. Velocity will start
at rest and evolve and add in mass dependency
"""
num_par = 100000
dt = 0.01
n = 1000
m=np.ones(num_par)
mk3 = k3_mass(num_par)
x = n*np.random.rand(num_par,2)
v = x*0 
g = greens2(n)
x,v = take_leapfrog_step(x, v,dt,mk3,n, num_par, g)
plt.show()
plt.clf() 



