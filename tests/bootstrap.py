#!/usr/bin/env python3

import numpy as np
from statistic import bootstrap
import statistic_tools as st

rng = np.random.default_rng(42)

data = rng.normal(loc=0.0, scale=0.1, size=(1000,))

smean, serr = bootstrap(data, n_boot=data.shape[0])
mean, err = st.bootstrap(data)

print(mean, smean)
print(err,  serr)

assert abs(smean-mean) < min([serr, err])*5
