#!/usr/bin/env python3

import numpy as np
import scipy.optimize as opt
import statistic_tools.general as st
import inspect


def chi2cost(x0, f, xdata, ydata, yerr):
    return (f(xdata, *x0) - ydata)/yerr


def single_chi2_fit(ydata, idxs, estimator=None, args={}):
    xdata = args['xdata']
    func = args['func']
    yerr = args['yerr']

    best_of = args.get('best_of', 10)

    nargs = len(str(inspect.signature(func)).split(','))-1
    sol = opt.least_squares(chi2cost, np.ones((nargs,)), method='lm',
                           args=(func, xdata[idxs], ydata[idxs], yerr[idxs]))

    chi2 = 2*sol.cost

    for i in range(best_of):
        x0 = np.random.uniform(0,10,size=nargs)
        sol_ = opt.least_squares(chi2cost, x0, method='lm', args=(func,
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

def chi2_fitting(f, xdata, ydata, yerr=[], args0=[], seed=42):

    chi2args = {'xdata': np.array(xdata),
                'yerr': np.array(yerr), 
                'func': f,
               }
    rng = np.random.default_rng(seed)
    vals, vals_err = st.bootstrap(ydata, sample_estimate_func=single_chi2_fit,
                                  estimator=None, args=chi2args,
                                  withoriginal=True, rng=rng)

    popt = vals[:-1]
    perr = vals_err[:-1]
    chi2 = (vals[-1], vals_err[-1])

    return popt, perr, chi2


