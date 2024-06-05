import math
import numpy as np
import scipy as sp
import sys
import pytest

from shangrla.core.NonnegMean import NonnegMean

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

    def test_lam_to_eta_to_lam(self):
        N = 100
        u = 1
        test = NonnegMean(N=N, u=u)
        for lam in [0.5, 1, 2]:
            for mu in [0.5, 0.7, 0.9]:
                eta = mu*(1+lam*(u-mu))
                np.testing.assert_almost_equal(test.lam_to_eta(lam, mu), eta)
                np.testing.assert_almost_equal(test.eta_to_lam(test.lam_to_eta(lam, mu), mu), lam)
        lam = np.array([0.5, 0.6, 0.7])
        mu = np.array([0.6, 0.65, 0.55])
        eta = np.array(list(mu[i]*(1+lam[i]*(u-mu[i])) for i in range(len(lam))))
        np.testing.assert_almost_equal(test.lam_to_eta(lam, mu), eta)
        np.testing.assert_almost_equal(test.eta_to_lam(test.lam_to_eta(lam, mu), mu), lam)

    def test_agrapa(self):
        t = 0.5
        c_g_0 = 0.5
        c_g_m = 0.99
        c_g_g = 0
        N = np.infty
        u = 1
        n=10
        # test for sampling with replacement, constant c
        for val in [0.6, 0.7]:
            for lam in [0.2, 0.5]:
                test = NonnegMean(N=N, u=u, bet=NonnegMean.fixed_bet,
                                  c_grapa_0=c_g_0, c_grapa_m=c_g_m, c_grapa_grow=c_g_g, 
                                  lam=lam)
                x = val*np.ones(n)
                lam_0 = test.agrapa(x)
                term = max(0, min(c_g_0/t, (val-t)/(val-t)**2))
                lam_t = term*np.ones_like(x)
                lam_t[0] = lam
                np.testing.assert_almost_equal(lam_0, lam_t)
        # test for sampling without replacement, growing c, but zero sample variance
        N = 10
        n = 5
        t = 0.5
        c_g_0 = 0.6
        c_g_m = 0.9
        c_g_g = 2
        for val in [0.75, 0.9]:
            for lam in [0.25, 0.5]:
                test = NonnegMean(N=N, u=u, bet=NonnegMean.agrapa,
                                  c_grapa_0=c_g_0, c_grapa_max=c_g_m, c_grapa_grow=c_g_g, 
                                  lam=lam)
                x = val*np.ones(n)
                lam_0 = test.agrapa(x)
                t_adj = np.array([(N*t - i*val)/(N-i) for i in range(n)])
                mj = val
                lam_t = (mj-t_adj)/(mj-t_adj)**2
                lam_t = np.insert(lam_t, 0, lam)[0:-1]
                j = np.arange(n)
                cj = c_g_0 + (c_g_m-c_g_0)*(1-1/(1+c_g_g*np.sqrt(j)))
                lam_t = np.minimum(cj/t_adj, lam_t)
                np.testing.assert_almost_equal(lam_0, lam_t)

    def test_betting_mart(self):
        N = np.infty
        n = 20
        t = 0.5
        u = 1
        for val in [0.75, 0.9]:
            for lam in [0.25, 0.5]:
                test = NonnegMean(N=N, u=u, bet=NonnegMean.fixed_bet, lam=lam)
                x = val*np.ones(n)
                np.testing.assert_almost_equal(test.betting_mart(x)[0], 1/(1+lam*(val-t))**n)

    def test_sjm(self):
        # test_sjm_with_replacement:
        test = NonnegMean()
        S, Stot, j, m = test.sjm(np.inf, 0.52, np.array([1, 0, 0.5, 4, 0.5]))
        np.testing.assert_array_equal(S, np.array([0, 1, 1, 1.5, 5.5]))
        assert Stot == 6
        np.testing.assert_array_equal(j, np.array([1, 2, 3, 4, 5]))
        np.testing.assert_array_equal(m, np.array([0.52, 0.52, 0.52, 0.52, 0.52]))

        # test_sjm_without_replacement:
        test = NonnegMean()
        S, Stot, j, m = test.sjm(5, 1.53, np.array([1, 2, 0.5, 1, 4]))
        np.testing.assert_array_equal(S, np.array([0, 1, 3, 3.5, 4.5]))
        assert Stot == 8.5
        np.testing.assert_array_equal(j, np.array([1, 2, 3, 4, 5]))
        np.testing.assert_array_almost_equal(m, np.array([1.53, 1.6625, 1.55, 2.075, 3.15]))

        # test_sjm_with_sample_larger_than_population:
        test = NonnegMean()
        with pytest.raises(AssertionError):
            test.sjm(4, 0.55, np.array([1, 2, 3, 4, 5]))

        # test_sjm_with_non_integer_population:
        test = NonnegMean()
        with pytest.raises(AssertionError):
            test.sjm(4.5, 0.56, np.array([1, 2, 3, 4, 5]))

##########################################################################################
if __name__ == "__main__":
    sys.exit(pytest.main(["-qq"], plugins=None))
