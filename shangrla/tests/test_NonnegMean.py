import math
import numpy as np
import scipy as sp
import pandas as pd
import json
import csv
import warnings
import typing
import sys
import pytest
import coverage
from numpy import testing
from collections import OrderedDict, defaultdict
from cryptorandom.cryptorandom import SHA256, random, int_from_hash
from cryptorandom.sample import random_permutation
from cryptorandom.sample import sample_by_index


from shangrla.Audit import Audit, Assertion, Assorter, Contest, CVR, Stratum
from shangrla.NonnegMean import NonnegMean
from shangrla.Dominion import Dominion
from shangrla.Hart import Hart

##########################################################################################
class TestNonnegMean:

    def test_alpha_mart(self):
        eps = 0.0001  # Generic small value

        # When all the items are 1/2, estimated p for a mean of 1/2 should be 1.
        s = np.ones(5)/2
        test = NonnegMean(N=int(10**6))
        np.testing.assert_almost_equal(test.alpha_mart(s)[0],1.0)
        test.t = eps
        np.testing.assert_array_less(test.alpha_mart(s)[1][1:],[eps]*(len(s)-1))

        s = [0.6,0.8,1.0,1.2,1.4]
        test.u=2
        np.testing.assert_array_less(test.alpha_mart(s)[1][1:],[eps]*(len(s)-1))

        s1 = [1, 0, 1, 1, 0, 0, 1]
        test.u=1
        test.N = 7
        test.t = 3/7
        alpha_mart1 = test.alpha_mart(s1)[1]
        # p-values should be big until the last, which should be 0
        print(f'{alpha_mart1=}')
        assert(not any(np.isnan(alpha_mart1)))
        assert(alpha_mart1[-1] == 0)

        s2 = [1, 0, 1, 1, 0, 0, 0]
        alpha_mart2 = test.alpha_mart(s2)[1]
        # Since s1 and s2 only differ in the last observation,
        # the resulting martingales should be identical up to the next-to-last.
        # Final entry in alpha_mart2 should be 1
        assert(all(np.equal(alpha_mart2[0:(len(alpha_mart2)-1)],
                            alpha_mart1[0:(len(alpha_mart1)-1)])))
        print(f'{alpha_mart2=}')

    def test_shrink_trunc(self):
        epsj = lambda c, d, j: c/math.sqrt(d+j-1)
        Sj = lambda x, j: 0 if j==1 else np.sum(x[0:j-1])
        tj = lambda N, t, x, j: (N*t - Sj(x, j))/(N-j+1) if np.isfinite(N) else t
        etas = [.51, .55, .6]  # alternative means
        t = 1/2
        u = 1
        d = 10
        f = 0
        vrand =  sp.stats.bernoulli.rvs(1/2, size=20)
        v = [
            np.array([0, 0, 0, 0, 0, 1, 1, 1, 1, 1]),
            np.array([1, 1, 1, 1, 1, 1, 0, 0, 0, 0]),
            vrand
        ]
        test_inf = NonnegMean(N=np.inf, t=t, u=u, d=d, f=f)
        test_fin = NonnegMean(          t=t, u=u, d=d, f=f)
        for eta in etas:
            c = (eta-t)/2
            test_inf.c = c
            test_inf.eta = eta
            test_fin.c=c
            test_fin.eta=eta
            for x in v:
                N = len(x)
                test_fin.N = N
                xinf = test_inf.shrink_trunc(x)
                xfin = test_fin.shrink_trunc(x)
                yinf = np.zeros(N)
                yfin = np.zeros(N)
                for j in range(1,N+1):
                    est = (d*eta + Sj(x,j))/(d+j-1)
                    most = u*(1-np.finfo(float).eps)
                    yinf[j-1] = np.minimum(np.maximum(t+epsj(c,d,j), est), most)
                    yfin[j-1] = np.minimum(np.maximum(tj(N,t,x,j)+epsj(c,d,j), est), most)
                np.testing.assert_allclose(xinf, yinf)
                np.testing.assert_allclose(xfin, yfin)

    def test_kaplan_markov(self):
        s = np.ones(5)
        test = NonnegMean(u=1, N=np.inf, t=1/2)
        np.testing.assert_almost_equal(test.kaplan_markov(s)[0], 2**-5)
        s = np.array([1, 1, 1, 1, 1, 0])
        test.g=0.1
        np.testing.assert_almost_equal(test.kaplan_markov(s)[0],(1.1/.6)**-5)
        test.random_order = False
        np.testing.assert_almost_equal(test.kaplan_markov(s)[0],(1.1/.6)**-5 * .6/.1)
        s = np.array([1, -1])
        try:
            test.kaplan_markov(s)
        except ValueError:
            pass
        else:
            raise AssertionError

    def test_kaplan_wald(self):
        s = np.ones(5)
        test = NonnegMean()
        np.testing.assert_almost_equal(test.kaplan_wald(s)[0], 2**-5)
        s = np.array([1, 1, 1, 1, 1, 0])
        test.g = 0.1
        np.testing.assert_almost_equal(test.kaplan_wald(s)[0], (1.9)**-5)
        test.random_order = False
        np.testing.assert_almost_equal(test.kaplan_wald(s)[0], (1.9)**-5 * 10)
        s = np.array([1, -1])
        try:
            test.kaplan_wald(s)
        except ValueError:
            pass
        else:
            raise AssertionError

    def test_sample_size(self):
        eta = 0.75
        u = 1
        N = 1000
        t = 1/2
        alpha = 0.05
        prefix = True
        quantile = 0.5
        reps=None
        prefix=False

        test = NonnegMean(test=NonnegMean.alpha_mart,
                              estim=NonnegMean.fixed_alternative_mean,
                              u=u, N=N, t=t, eta=eta)

        x = np.ones(math.floor(N/200))
        sam_size = test.sample_size(x=x, alpha=alpha, reps=reps, prefix=prefix, quantile=quantile)
        np.testing.assert_equal(sam_size, 8) # ((.75/.5)*1+(.25/.5)*0)**8 = 25 > 1/alpha, so sam_size=8
    #
        reps=100
        sam_size = test.sample_size(x=x, alpha=alpha, reps=reps, prefix=prefix, quantile=quantile)
        np.testing.assert_equal(sam_size, 8) # all simulations should give the same answer
    #
        x = 0.75*np.ones(math.floor(N/200))
        sam_size = test.sample_size(x=x, alpha=alpha, reps=reps, prefix=prefix, quantile=quantile)
        np.testing.assert_equal(sam_size, 14) # ((.75/.5)*.75+(.25/.5)*.25)**14 = 22.7 > 1/alpha, so sam_size=14
    #
        g = 0.1
        x = np.ones(math.floor(N/200))

        test = NonnegMean(test=NonnegMean.kaplan_wald,
                              estim=NonnegMean.fixed_alternative_mean,
                              u=u, N=N, t=t, eta=eta, g=g)
        sam_size = test.sample_size(x=x, alpha=alpha, reps=None, prefix=prefix, quantile=quantile)
    #   p-value is \prod ((1-g)*x/t + g), so
        kw_size = math.ceil(math.log(1/alpha)/math.log((1-g)/t + g))
        np.testing.assert_equal(sam_size, kw_size)

        x = 0.75*np.ones(math.floor(N/200))
        sam_size = test.sample_size(x=x, alpha=alpha, reps=None, prefix=prefix, quantile=quantile)
    #   p-value is \prod ((1-g)*x/t + g), so
        kw_size = math.ceil(math.log(1/alpha)/math.log(0.75*(1-g)/t + g))
        np.testing.assert_equal(sam_size, kw_size)
        
##########################################################################################
if __name__ == "__main__":
    sys.exit(pytest.main(["-qq"], plugins=None))
