import numpy as np
import camb



def get_spectrum(pars,lmax=2000):
    #print('pars are ',pars)
    H0=pars[0]
    ombh2=pars[1]
    omch2=pars[2]
    tau=pars[3]
    As=pars[4]
    ns=pars[5]
    pars=camb.CAMBparams()
    pars.set_cosmology(H0=H0,ombh2=ombh2,omch2=omch2,mnu=0.06,omk=0,tau=tau)
    pars.InitPower.set_params(As=As,ns=ns,r=0)
    pars.set_for_lmax(lmax,lens_potential_accuracy=0)
    results=camb.get_results(pars)
    powers=results.get_cmb_power_spectra(pars,CMB_unit='muK')
    cmb=powers['total']
    tt=cmb[:,0]    #you could return the full power spectrum here if you wanted to do say EE
    return tt

pars=np.asarray([65,0.02,0.1,0.05,2e-9,0.96])

x = np.array(get_spectrum(pars))
cmb = x[2:1201] #cut out values after 1200 term since wmap is short
data=np.loadtxt('wmap_tt_spectrum_9yr_v5.txt')
wmap = data[:,1]
error = data[:,2]

##Solve for chi^2 now (x-m)^2/error
chisq = (wmap - cmb)**2/error**2
print(np.sum(chisq))

