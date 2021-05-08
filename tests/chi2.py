#!/usr/bin/env python3

import numpy as np
import statistic_tools.fitting as f
import matplotlib.pyplot as plt
from statistic_tools.general import bootstrap
# Define function

def func(x, A, E):
    return A*np.exp(-E*x)

# Global variables
Nd = 20            # Number of samples
A  = 1.0
E  = 2.3


# Setup random environment
rng = np.random.default_rng(42)

xvals = np.linspace(0,1,10)
_yvals = func(xvals, A, E)

# Add random noise to the samples
yvals = np.zeros((Nd, xvals.shape[0]), dtype=np.double)

for i in range(Nd):
    yvals[i,:] = _yvals + rng.normal(loc=0.0, scale=0.05*xvals+0.01,
                                     size=_yvals.shape[0])

# Create plot of the sample
fig, ax = plt.subplots(1,1,dpi=150,figsize=(8,5))
_xvals = np.linspace(0,1,100)
ax.plot(_xvals, func(_xvals,A,E),lw=1,color='tab:red')
for i in range(Nd):
    ax.plot(xvals,yvals[i,:],lw=.1,color='black',marker='x',markersize=3)
    plt.savefig('chi2_sample_plot.pdf',format='pdf')


# Find parameter which minimize the chi^2 function
param0 = np.array([2.0, 3.0])
fitter = f.Chi2fitter(xvals,yvals,func,param0)
param, errs, _ = fitter.evaluate(verbose=True)

assert abs(param[0]-A)<errs[0]
assert abs(param[1]-E)<errs[1]

dt=0.1
fig, ax = plt.subplots(1,1,figsize=(8,5),dpi=150)
eff_mass, err_mass = bootstrap(np.log((yvals[:,:-1]+0*1j)/yvals[:,1:]).real)
ax.errorbar(xvals[:-1],eff_mass,yerr=err_mass,lw=.1,marker='s',markersize=3,
            elinewidth=1,capsize=3)
nxvals = xvals[:-1].shape[0]
ax.fill_between(xvals[:-1], [(param[1]+errs[1])*dt]*nxvals,
                [(param[1]-errs[1])*dt]*nxvals, alpha=.2, color='tab:blue')
ax.plot(xvals[:-1],[param[1]*dt]*nxvals,lw=1,ls='--',color='tab:blue')
plt.savefig('chi2_sample_mass_estimate.pdf',format='pdf')

