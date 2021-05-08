import numpy as np

def sample_estimate(data, sample_idxs, estimator=np.mean):
    return estimator([data[idx] for idx in sample_idxs], axis=0)


def bootstrap(data, estimator=np.mean, N=None, rng=None, seed=42):
    if rng == None:
        rng = np.random.default_rng(seed)

    if N == None:
        N = data.shape[0]

    idxs = range(data.shape[0])

    estimates = np.empty((N,),dtype=type(data[0]))
    for i in range(N):
        estimates[i] = sample_estimate(data,
                                       rng.choice(idxs, size=(data.shape[0],),
                                                  replace=True)
                                      )

    return np.mean(estimates, axis=0), np.std(estimates, ddof=1, axis=0)


def binning(data, nbins=1, estimator=np.mean):
    bin_size = data.shape[0]//nbins
    binned_data = np.empty((nbins,),dtype=type(data[0]))
    for i in range(nbins):
        if i != nbins-1:
            binned_data[i]=estimator(data[i*bin_size:(i+1)*bin_size],axis=0)
        else:
            # To prevent index overflow
            binned_data[i]=estimator(data[i*bin_size:],axis=0)
    return binned_data





