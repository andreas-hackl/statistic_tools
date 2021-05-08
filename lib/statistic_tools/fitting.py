#!/usr/bin/env python3

import numpy as np
import scipy.optimize as opt
from statistic_tools.general import bootstrap
import time

def cost(param, index, chifitter):
    xdata = chifitter.xdata
    ydata = chifitter.ydata
    ffunc = chifitter.ffunc
    nmin = chifitter.nmin
    nmax = chifitter.nmax
    W = chifitter.W

    return np.sum([(ydata[index, ni]-ffunc(xdata[ni], *param))*W[ni,nj]*
                    (ydata[index, nj]-ffunc(xdata[nj], *param))
                    for ni in range(nmin,nmax+1) for nj in range(nmin,nmax+1)])


class Chi2fitter:

    def __init__(self, xdata, ydata, ffunc, param0, nmin=0, nmax=-1, tol=1e-6,
                 W=None, opt_method='BFGS', options={}):
        if xdata.shape[0] != ydata.shape[1]:
            raise ValueError("first dim of xdata and ydata are expected to be"
                             +f"the same, got {xdata.shape} and {ydata.shape}")
        self.xdata = xdata
        self.ydata = ydata
        self.Nd, self.Nt = ydata.shape

        self.ffunc = ffunc
        self.param0 = param0
        self.param = param0

        if 0 <= nmin < self.Nt:
            self.nmin = nmin
        else:
            raise ValueError(f"nmin have to lie between 0 and Nt-1, got {nmin}")

        if nmin < nmax < self.Nt:
            self.nmax = nmax
        elif nmax == -1:
            self.nmax = self.Nt-1
        else:
            raise ValueError(f"nmax have to lie between nmin and Nt-1, got {nmax}")

        if W == None:
            self.W = self._calc_cov()
        else:
            if W.shape != (self.Nt, self.Nt):
                raise ValueError(f"W has the wrong shape, expected ({self.Nt},{self.Nt}), got {W.shape}")
            else:
                self.W = W

        self.tol = tol
        self.opt_method='Nelder-Mead'
        self.options=options

    def _calc_cov(self):
        means = np.mean(self.ydata, axis=0)
        print(means)

        W = np.zeros((self.Nt, self.Nt), dtype=np.double)
        for ni in range(self.Nt):
            for nj in range(ni, self.Nt):
                tmp = 1/self.Nd *  np.mean([(self.ydata[i, ni]-means[ni])*
                                            (self.ydata[i, nj]-means[nj])
                                            for i in range(self.Nd)
                                           ])

                W[ni, nj] = tmp**(-1)
                W[nj, ni] = tmp**(-1)

        return W

    def _single_index_fit(self, index):

        param = self.param0

        sol = opt.minimize(cost, param, args=(index, self),
                           method=self.opt_method, tol=self.tol,
                           options=self.options)

        return sol.x, sol.fun

    def evaluate(self, verbose=True):
        vals = np.zeros((self.Nd, self.param.shape[0]), dtype=np.double)
        funs = np.zeros((self.Nd, ), dtype=np.double)
        if verbose:
            print('Start fitting ...')
        TIME_START = time.time()

        for index in range(self.Nd):
            if verbose:
                print(f'sample {index+1:5d}/{self.Nd:d}', end='\t')
            param, fun = self._single_index_fit(index)
            if verbose: print(f'{time.time()-TIME_START:6.3f} [sec]')
            vals[index,:] = param
            funs[index] = fun

        params, params_err = bootstrap(vals)
        av_fit_error = np.mean(funs)

        if verbose:
            print(f'DONE IN {time.time()-TIME_START:6.3f} [sec]')
            return params, params_err, av_fit_error

        return params, params_err
