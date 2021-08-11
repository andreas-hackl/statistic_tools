import numpy as np
import time

def sample_estimate(data, sample_idxs, estimator=np.mean, args=None):
    sampled_data = np.array([data[idx] for idx in sample_idxs])
    try:
        return estimator(sampled_data, axis=0, args=args)
    except TypeError:
        return estimator(sampled_data, axis=0)


def bootstrap(data, estimator=np.mean, N=None, rng=None, seed=None, args=None,
             sample_estimate_func=sample_estimate, withoriginal=False,
              sample_size=None):
    if seed == None:
        seed = int(time.time())

    if rng == None:
        rng = np.random.default_rng(seed)

    if N == None:
        N = data.shape[0]

    if sample_size == None:
        sample_size = data.shape[0]

    idxs = range(data.shape[0])

    estimates = []
    if withoriginal:
        estimate = sample_estimate_func(data, idxs, estimator=estimator,
                                        args=args)
        estimates.append(estimate)

    for i in range(N):
        estimate = sample_estimate_func(data,
                                        rng.choice(idxs, size=(sample_size,),
                                                  replace=True),
                                        estimator=estimator, args=args,
                                        )
        estimates.append(estimate)

    estimates = np.array(estimates)

    if data.dtype == "complex":
        mean_data = np.mean(estimates, axis=0)
        std_data = np.std(estimates.real, ddof=1, axis=0) + 1j*np.std(estimates.imag, ddof=1, axis=0)
        return mean_data, std_data

    return np.mean(estimates, axis=0), np.std(estimates, ddof=1, axis=0)

def bootstrap_samples(data, estimator=np.mean, N=None, rng=None, seed=None,
                      args=None):
    if seed == None:
        seed = int(time.time())

    if rng == None:
        rng = np.random.default_rng(seed)

    if N == None:
        N = data.shape[0]

    idxs = range(data.shape[0])

    estimates = np.empty((N,), dtype=type(data[0]))

    for i in range(N):
        estimates[i] = sample_estimate(data,
                                       rng.choice(idxs, size=(data.shape[0],),
                                                  replace=True),
                                       estimator=estimator, args=args
                                      )
    return estimates


def binning(data, nbins=1, estimator=np.mean):
    bin_size = data.shape[0]//nbins
    binned_data = []
    for i in range(nbins):
        if i != nbins-1:
            binned_data.append(estimator(data[i*bin_size:(i+1)*bin_size],axis=0))
        else:
            # To prevent index overflow
            binned_data.append(estimator(data[i*bin_size:],axis=0))
    return np.array(binned_data)





