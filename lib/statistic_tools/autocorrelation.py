#!/usr/bin/env python3

import numpy as np


class Autocorrelation:
    def __init__(self, data):
        self.data = data
        self.R0 = self._auto_covariance(0)
        self.autocov_status = False

    def _auto_covariance(self, tau):
        N = self.data.shape[0]
        d = self.data
        _m = np.mean(d, axis=0)
        return 1/(N-tau)*np.sum([(d[i]-_m)*(d[i+tau]-_m) for i in
                                 range(N-tau)], axis=0)

    def eval(self, tau=[]):
        if isinstance(tau, (float, int)):
            return self._auto_covariance(tau)/self.R0
        if len(tau) == 0:
            tau = np.array(range(self.data.shape[0]))
        self.autocov = np.array(
                                [self._auto_covariance(t)/self.R0 for t in tau]
                               )
        self.autocov_status = True
        return self.autocov

    def int_auto_correlation_time(self):
        if self.autocov_status == False:
            self.eval()
        ac = self.autocov
        val = 0.5
        idx = 1
        while ac[idx] > 0:
            val += ac[idx]
            idx += 1
        return val

    def tau(self):
        return self.int_auto_correlation_time()
