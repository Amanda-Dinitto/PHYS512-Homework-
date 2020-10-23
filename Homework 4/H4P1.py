
import numpy as np
from matplotlib import pyplot as plt

def convolution(f, g):
    F = np.fft.fft(f)
    G = np.fft.fft(g)
    H = np.fft.ifft(F*G)
    return H 

x = np.arange(-20,20,0.1)
dx1 = 0 ##no shifting
dx2 = 10 ## with shifting 
def gaussian(x, dx): 
    a = x+dx 
    func = np.exp(-0.5*(a)**2/(0.75**2))
    return func

func1 = gaussian(x, dx1) ##original gaussian before shifting 
f = func1/func1.sum()
func2 = gaussian(x, dx2)
g = 1.43*func2/func2.sum() ##multiplied by factor so convolution is same height as original gaussian
H = np.real(convolution(f, g))
plt.plot(x, f)
#plt.plot(x,g)
plt.plot(x, H)
plt.title('Gaussian Convolved and Shifted')
plt.savefig('H4P1.png')
plt.show()
    
