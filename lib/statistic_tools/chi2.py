#!/usr/bin/env python3

import numpy as np
import scipy.optimize as opt
import statistic_tools.general as st
import inspect


def cost(x0, f, xdata, ydata, yerr):
    return (f(xdata, *x0) - ydata)/yerr


def single_fit(ydata, idxs, estimator=None, args={}):
    xdata = args['xdata']
    func = args['func']
    yerr = args['yerr']
    rng = args['rng']

    best_of = args.get('best_of', 3)

    nargs = len(str(inspect.signature(func)).split(','))-1
    sol = opt.least_squares(cost, np.ones((nargs,)), method='lm',
                           args=(func, xdata[idxs], ydata[idxs], yerr[idxs]))

    chi2 = 2*sol.cost

    for i in range(best_of):
        x0 = np.random.uniform(0,10,size=nargs)
        sol_ = opt.least_squares(cost, x0, method='lm', args=(func,
                                                              xdata[idxs],
                                                              ydata[idxs],
                                                              yerr[idxs])
                                )

        chi2_ = 2*sol_.cost

        if chi2_ < chi2:
            chi2 = chi2_
            sol = sol_

    param = list(sol.x) + [chi2]
    return param

def bootstrap_fitting(f, xdata, ydata, yerr=[], args0=[], seed=42, rng=None,
                      Nrun=10):
    if rng == None:
        rng = np.random.default_rng(seed)

    if len(yerr) == 0:
        yerr = np.ones_like(ydata)

    chi2args = {'xdata': np.array(xdata),
                'yerr': np.array(yerr),
                'func': f,
                'rng': rng,
               }
    vals, vals_err = st.bootstrap(ydata, sample_estimate_func=single_fit,
                                  estimator=None, args=chi2args,
                                  withoriginal=True, rng=rng,
                                  N=xdata.shape[0]*Nrun)

    popt = vals[:-1]
    perr = vals_err[:-1]
    chi2 = (vals[-1], vals_err[-1])

    return popt, perr, chi2

class DataSample:

    def __init__(self, N, mus, sigmas, seed=42):
        self.N = N
        self.sample = np.zeros((N, ), dtype=np.double)
        self.mus = np.array(mus)
        self.sigmas = np.array(sigmas)
        self.rng = np.random.default_rng(seed)

    def sweep(self):
        for k in range(self.N):
            yki = self.sample[k]
            sigk = self.sigmas[k]
            mk = self.mus[k]

            dyki = self.rng.normal(loc=0.0, scale=sigk)

            dChi2 = (2*(yki - mk)*dyki + dyki**2)/(sigk**2)

            if dChi2 < 0:
                self.sample[k] += dyki

            else:
                P = np.exp(-1/2 * dChi2)
                q = self.rng.uniform(0,1)
                if q < P:
                    self.sample[k] += dyki

    def chi2(self):
        return sum([((y - m)/s)**2 for y, m, s in zip(self.sample, self.mus,
                                                      self.sigmas)])


def mc_fitting_old(f, xdata, ydata, yerr=[], args0=[], seed=42, rng=None, Nrun=10):

    chi2vals = []
    _popt = []
    yerr = np.ones_like(ydata)

    data = DataSample(xdata.shape[0], ydata, yerr, seed=seed)
    # Thermalization
    for i in range(100):
        data.sweep()
        chi2vals.append(data.chi2())

    for i in range(Nrun):
        for k in range(10):
            data.sweep()
            chi2vals.append(data.chi2())

        nargs = len(str(inspect.signature(f)).split(','))-1
        sol = opt.least_squares(cost, np.random.uniform(0, 1, size=nargs), method='lm',
                                args=(f, xdata, data.sample, yerr))

        _popt.append(sol.x)
    _popt = np.array(_popt)
    popt, perr = st.bootstrap(_popt)

    return popt, perr, np.array(chi2vals)

def get_param(f, xdata, ydata, yerr, nargs, rng, best_of=3):
    sol = None
    chi2val = 1e8

    for i in range(best_of):
        p0 = rng.uniform(0, 10, size=nargs)
        sol_ = opt.least_squares(cost, p0, method='lm', args=(f, xdata, ydata,
                                                              yerr))
        chi2val_ = 2*sol_.cost
        if chi2val_ < chi2val:
            sol = sol_
            chi2val = chi2val_

    return sol.x, chi2val



def mc_fitting(f, xdata, ydata, yerr=[], args0=[], seed=42, rng=None, Nrun=10):
    if rng == None:
        rng = np.random.default_rng(seed)

    if len(yerr) == 0:
        yerr = np.ones_like(ydata)

    nargs = len(str(inspect.signature(f)).split(','))-1

    popts = []
    chivals = []
    # Generate pseudo_sample

    for i in range(Nrun):
        ydata_pseudo = ydata + [rng.normal(loc=0.0, scale=ye) for ye in yerr]

        _popt, chi2val = get_param(f, xdata, ydata_pseudo, yerr, nargs, rng)
        popts.append(_popt)
        chivals.append(chi2val)
    popts = np.array(popts)
    chivals = np.array(chivals)
    chi2val = st.bootstrap(chivals)
    popt, perr = st.bootstrap(popts)
    return popt, perr, chi2val



def fitting(f, xdata, ydata, yerr=[], args0=[], seed=42, N=10, rng=None,
            method='bootstrap'):
    xdata = np.array(xdata)
    ydata = np.array(ydata)
    yerr = np.array(yerr)
    if method=='bootstrap':
        return bootstrap_fitting(f, xdata, ydata, yerr=yerr, args0=args0,
                                 seed=seed, Nrun=N, rng=rng)

    if method=='mc':
        return mc_fitting(f, xdata, ydata, yerr=yerr, args0=args0, seed=seed,
                          rng=rng, Nrun=N)

    raise ValueError(f'Invalid method, got {method}')
