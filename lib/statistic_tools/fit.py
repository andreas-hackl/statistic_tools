#!/usr/bin/env python3

import numpy as np
import optilib.lm as lm
from optilib.grad import jacobian

def fit_y(f, p0, xdata, ydata, yerr, verbose=False, **kwargs):
    model = lambda p: f(xdata, p)
    W = np.diag(1/yerr**2)
    lsqf = lm.LSQ_fit(p0, model, ydata, W,
                      maxiter=kwargs.get('maxiter', 1000),
                      eps1=kwargs.get('eps1', 1e-3),
                      eps2=kwargs.get('eps2', 1e-3),
                      eps3=kwargs.get('eps3', 1e-1),
                      eps4=kwargs.get('eps4', 1e-1),
                      alpha0=kwargs.get('alpha0', 0.01),
                      eps_grad=kwargs.get('eps_grad', 1e-9),
                      Aup=kwargs.get('Aup', 11),
                      Adown=kwargs.get('Adown', 9),
                     )

    results = lsqf.solve(verbose=verbose, mode=kwargs.get('method', 'lm'))
    return results

def fit_xy(f, p0, xdata, ydata, xerr, yerr, verbose=False, **kwargs):
    def model_vec(p, f, xdata, nparam):
        n = xdata.shape[0]
        dx = p[nparam:]
        beta = p[:nparam]

        vec = np.zeros((2*n,), dtype=np.double)
        vec[:n] = f(xdata + dx, p)
        vec[n:] = dx

        return vec

    nparam = len(p0)

    print(kwargs.get('maxiter', 1))

    beta0 = np.concatenate((p0, np.zeros_like(xdata)))
    model = lambda p: model_vec(p, f, xdata, nparam)
    W = np.diag(np.concatenate((1/yerr**2, 1/xerr**2)))
    ydata_ = np.concatenate((ydata, np.zeros_like(xdata)))
    lsqf = lm.LSQ_fit(beta0, model, ydata_, W,
                      maxiter=kwargs.get('maxiter', 1000),
                      eps1=kwargs.get('eps1', 1e-3),
                      eps2=kwargs.get('eps2', 1e-3),
                      eps3=kwargs.get('eps3', 1e-1),
                      eps4=kwargs.get('eps4', 1e-1),
                      alpha0=kwargs.get('alpha0', 0.01),
                      eps_grad=kwargs.get('eps_grad', 1e-9),
                      Aup=kwargs.get('Aup', 11),
                      Adown=kwargs.get('Adown', 9),
                     )
    return lsqf.solve(verbose=verbose, mode=kwargs.get('method', 'lm'))


def fit(f, p0, xdata, ydata, yerr, xerr=None, verbose=False, full_output=False, **kwargs):
    if xerr is not None:
        results = fit_xy(f, p0, xdata, ydata, xerr, yerr, verbose=verbose, **kwargs)
    else:
        results = fit_y(f, p0, xdata, ydata, yerr, verbose=verbose, **kwargs)

    if verbose:
        print("RESULT:\n"+'='*20+'\n')
        for name, val in results.items():
            print(f"{name}:\t{val}")

    if full_output: return results

    popt = results['beta']
    pcov = results['cov']

    if len(popt) > len(p0):
        nparam = len(p0)
        popt = popt[:nparam]
        pcov = pcov[:nparam,:nparam]
    return popt, pcov


def fit_multiple(f, xdata_array, ydata_array, yerr_array, verbose=False, **kwargs):

    def model_vec(p, f, xdata_array):
        n = len(xdata_array)
        Nis = [xdata.shape[0] for xdata in xdata_array]
        Nis_cum = [int(np.sum(Nis[:i])) for i in range(0, len(Nis)+1)]

        N = Nis_cum[-1]

        E = p[0]
        dE = p[1]
        As = p[2:]

        vec = np.zeros((N, ), dtype=np.double)

        for i in range(n):
            vec[Nis_cum[i]:Nis_cum[i+1]] = f(xdata_array[i], [E, dE, As[i]])

        return vec


    assert len(xdata_array) == len(ydata_array)
    assert len(xdata_array) == len(yerr_array)

    nf = len(xdata_array)

    W = np.diag(np.concatenate(([1/yerr**2 for yerr in yerr_array])))
    ydata_ = np.concatenate(ydata_array)

    beta0 = np.ones((2+nf), dtype=np.double)

    model = lambda p: model_vec(p, f, xdata_array)

    lsqf = lm.LSQ_fit(beta0, model, ydata_, W, 
                      maxiter=kwargs.get('maxiter', 1000),
                      eps1=kwargs.get('eps1', 1e-3),
                      eps2=kwargs.get('eps2', 1e-3),
                      eps3=kwargs.get('eps3', 1e-1),
                      eps4=kwargs.get('eps4', 1e-1),
                      alpha0=kwargs.get('alpha0', 0.01),
                      eps_grad=kwargs.get('eps_grad', 1e-9),
                      Aup=kwargs.get('Aup', 11),
                      Adown=kwargs.get('Adown', 9),
                     )

    sol = lsqf.solve(verbose=verbose, mode=kwargs.get('method', 'lm'))

    popts = []
    pcovs = []


    for i in range(nf):
        mask = np.zeros((2+nf, ), dtype=bool)
        mask[:2] = True
        mask[i+2] = True

        popts.append(sol['beta'][mask])

        mask2d = np.outer(mask, mask)
        pcov = sol['cov'][mask2d].reshape((3,3))
        pcovs.append(pcov)


    if verbose:
        return popts, pcovs, sol
    return popts, pcovs




def errorband(f, p0, xfine, pcov):
    jac = jacobian(lambda p: f(xfine, p), p0)
    fcov = np.dot(jac, np.dot(pcov, np.transpose(jac)))
    return np.sqrt(np.diag(fcov))
