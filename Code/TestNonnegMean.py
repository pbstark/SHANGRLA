import math
import numpy as np
import scipy as sp
from scipy.stats import bernoulli
import pandas as pd
import json
import csv
import warnings
import typing
from numpy import testing
from collections import OrderedDict, defaultdict
from cryptorandom.cryptorandom import SHA256, random, int_from_hash
from cryptorandom.sample import random_permutation
from cryptorandom.sample import sample_by_index

class TestNonnegMean:
    '''
    Tests of the hypothesis that the mean of a non-negative population is less than t.
        Several tests are implemented, all ultimately based on martingales or supermartingales:
            Kaplan-Markov
            Kaplan-Wald
            Kaplan-Kolmogorov
            Wald SPRT with replacement (only for binary-valued populations)
            Wald SPRT without replacement (only for binary-valued populations)
            Kaplan's martingale (KMart)
            ALPHA supermartingale test
    '''

    TESTS = ['kaplan_markov','kaplan_wald','kaplan_kolmogorov','wald_sprt','alpha_mart']

    @classmethod
    def wald_sprt(cls, x: np.array, N: int, t: float=1/2) -> typing.Tuple[float, np.array]:
        '''
        Finds the p value for the hypothesis that the population
        mean is less than or equal to t against the alternative that it is eta,
        for a population of size N of values in the interval [0, u].

        Generalizes Wald's SPRT for the Bernoulli to sampling without replacement and to
        nonnegative bounded values rather than binary values (for which it is a valid test, but not the SPRT).
        See Stark, 2022. ALPHA: Audit that Learns from Previous Hand-Audited Ballots

        If N is finite, assumes the sample is drawn without replacement
        If N is infinite, assumes the sample is with replacement

        If the sample is drawn without replacement, the data must be in random order

        Parameters
        ----------
        x: binary list, one element per draw. A list element is 1 if the
            the corresponding trial was a success
        N: int
            population size for sampling without replacement, or np.infinity for
            sampling with replacement
        t: float in (0,u)
            hypothesized population mean
        kwargs:
            eta: float in (0,u)
                alternative hypothesized population mean
            random_order: Boolean
                if the data are in random order, setting this to True can improve the power.
                If the data are not in random order, set to False

        Returns
        -------
        p: float
            p-value
        p_history: numpy array
            sample by sample history of p-values. Not meaningful unless the sample is in random order.
        '''
        u = kwargs.get('u', 1)
        eta = kwargs.get('eta', u*(1-np.finfo(float).eps))
        random_order = kwargs.get('random_order', True)
        if any((xx < 0 or xx > u) for xx in x):
            raise ValueError(f'Data out of range [0,{u}]')
        if np.isfinite(N):
            if not random_order:
                raise ValueError("data must be in random order for samples without replacement")
            S = np.insert(np.cumsum(x),0,0)[0:-1]  # 0, x_1, x_1+x_2, ...,
            j = np.arange(1,len(x)+1)              # 1, 2, 3, ..., len(x)
            m = (N*t-S)/(N-j+1)                    # mean of population after (j-1)st draw, if null is true
        else:
            m = t
        with np.errstate(divide='ignore',invalid='ignore'):
            terms = np.cumprod((x*eta/m + (u-x)*(u-eta)/(u-m))/u) # generalization of Bernoulli SPRT
        terms[m<0] = np.inf                        # the null is surely false
        terms = np.cumprod(terms)
        return 1/np.max(terms) if random_order else 1/terms[-1], np.minimum(1,1/terms)

    @classmethod
    def fixed_alternative_mean(cls, x: np.array, N: int, t: float=1/2, **kwargs) -> np.array:
        '''
        Compute the alternative mean just before the jth draw, for a fixed alternative that the original population 
        mean is eta.
        Throws a warning if the sample implies that the fixed alternative is false (because the population would
        have negative values or values greater than u.
        
        S_1 := 0
        S_j := \sum_{i=1}^{j-1} x_i, j >= 1
        eta_j := (N*eta-S_j)/(N-j+1) if np.isfinite(N) else t

        Parameters
        ----------
        x: np.array
            input data
        t: float in (0, 1)
            hypothesized population mean; not used
        kwargs:
            eta: float in (t, u) (default u*(1-eps))
                alternative hypothethesized value for the population mean
            u: float > 0 (default 1)
                upper bound on the population values
        '''
        u = kwargs.get('u', 1)
        eta = kwargs.get('eta', u*(1-np.finfo(float).eps))
        S = np.insert(np.cumsum(x),0,0)[0:-1]  # 0, x_1, x_1+x_2, ...,
        j = np.arange(1,len(x)+1)              # 1, 2, 3, ..., len(x)
        m = (N*eta-S)/(N-j+1) if np.isfinite(N) else t   # mean of population after (j-1)st draw, if eta is the mean
        if (negs := np.sum(m<0)) > 0:
            warnings.warn(f'Implied population mean is negative in {negs} of {len(x)} terms')
        if (pos := np.sum(m>u)) > 0:
            warnings.warn(f'Implied population mean is greater than {u=} in {pos} of {len(x)} terms')
        return m
    

    @classmethod
    def shrink_trunc(cls, x: np.array, N: int, t: float=1/2, **kwargs) -> np.array:
        '''
        apply shrinkage/truncation estimator to an array to construct a sequence of "alternative" values

        sample mean is shrunk towards eta, with relative weight d compared to a single observation,
        then that combination is shrunk towards u, with relative weight f/(stdev(x)).

        The result is truncated above at u-u*eps and below at m_j+e_j(c,j)

        Shrinking towards eta stabilizes the sample mean as an estimate of the population mean.
        Shrinking towards u takes advantage of low-variance samples to grow the test statistic more rapidly.

        The running standard deviation is calculated using Welford's method.

        S_1 := 0
        S_j := \sum_{i=1}^{j-1} x_i, j >= 1
        m_j := (N*t-S_j)/(N-j+1) if np.isfinite(N) else t
        e_j := c/sqrt(d+j-1)
        sd_1 := sd_2 = 1
        sd_j := sqrt[(\sum_{i=1}^{j-1} (x_i-S_j/(j-1))^2)/(j-2)] \wedge minsd, j>2
        eta_j :=  ( [(d*eta + S_j)/(d+j-1) + f*u/sd_j]/(1+f/sd_j) \vee (m_j+e_j) ) \wedge u*(1-eps)

        Parameters
        ----------
        x: np.array
            input data
        N: int
            population size
        t: float in (0, 1)
            hypothesized population mean
        kwargs: keyword arguments
            u: float > 0 (default 1)
                upper bound on the population values
            eta: float in (t, u) (default u*(1-eps))
                initial alternative hypothethesized value for the population mean
            c: positive float
                scale factor for allowing the estimated mean to approach t from above
            d: positive float
                relative weight of eta compared to an observation, in updating the alternative for each term
            f: positive float
                relative weight of the upper bound u (normalized by the sample standard deviation)
            minsd: positive float
                lower threshold for the standard deviation of the sample, to avoid divide-by-zero errors and
                to limit the weight of u
                
        '''
        # set the parameters
        c = kwargs.get('c', 1/2)
        d = kwargs.get('d', 100)
        f = kwargs.get('f', 0)
        u = kwargs.get('u', 1) 
        eta = kwargs.get('eta', u*(1-np.finfo(float).eps))      
        minsd = kwargs.get('minsd', 10**-6)
        #
        S = np.insert(np.cumsum(x),0,0)[0:-1]  # 0, x_1, x_1+x_2, ...,
        j = np.arange(1,len(x)+1)              # 1, 2, 3, ..., len(x)
        m = (N*t-S)/(N-j+1) if np.isfinite(N) else t   # mean of population after (j-1)st draw, if null is true
        # Welford's algorithm for running mean and running sd
        mj = [x[0]]                        
        sdj = [0]
        for i, xj in enumerate(x[1:]):
            mj.append(mj[-1]+(xj-mj[-1])/(i+1))
            sdj.append(sdj[-1]+(xj-mj[-2])*(xj-mj[-1]))
        sdj = np.sqrt(sdj/j)
        # end of Welford's algorithm. 
        # threshold the sd, set first two sds to 1
        sdj = np.insert(np.maximum(sdj,minsd),0,1)[0:-1] 
        sdj[1]=1
        weighted = ((d*eta+S)/(d+j-1) + u*f/sdj)/(1+f/sdj)
        return np.minimum(u*(1-np.finfo(float).eps), np.maximum(weighted,m+c/np.sqrt(d+j-1)))

    @classmethod
    def optimal_comparison(cls, x: np.array, N: int, t: float=1/2, **kwargs) -> np.array:
        '''
        The "bet" that is optimal for ballot-level comparison audits, for which overstatement
        assorters take a small number of possible values and (when the system behaved correctly)
        are concentrated on a single value.

        Parameters
        ----------
        x: np.array
            input data
        N: int
            population size (np.inf for sampling with replacement)
        t: float in (0, 1)
            hypothesized population mean
        kwargs: keyword arguments
            u: float > 0 (default 1)
                upper bound on the population values
            eta: float in (t, u) (default u*(1-eps))
                initial alternative hypothethesized value for the population mean
                
        '''
        # set the parameters
        u = kwargs.get('u', 1) 
        eta = kwargs.get('eta', u*(1-np.finfo(float).eps))      
        #
        raise(NotImplementedError)

    @classmethod
    def alpha_mart(cls, x: np.array, N: int, t: float=1/2, estim: callable=None, **kwargs) -> typing.Tuple[float, np.array] :
        '''
        Finds the ALPHA martingale for the hypothesis that the population
        mean is less than or equal to t using a martingale method,
        for a population of size N, based on a series of draws x.

        **The draws must be in random order**, or the sequence is not a supermartingale under the null

        If N is finite, assumes the sample is drawn without replacement
        If N is infinite, assumes the sample is with replacement

        Parameters
        ----------
        x: list corresponding to the data
        N: int
            population size for sampling without replacement, or np.infinity for sampling with replacement
        t: float in [0,u)
            hypothesized fraction of ones in the population
        estim: function (note: class methods are not of type Callable)
            estim(x, N, t, **kwargs) -> np.array of length len(x), the sequence of values of eta_j for ALPHA
        kwargs:
            keyword arguments for estim() and for this function
            u: float > 0 (default 1)
                upper bound on the population
            eta: float in (t,u] (default u*(1-eps))
                value parametrizing the bet. Use alternative hypothesized population mean for polling audit
                or a value nearer the upper bound for comparison audits


        Returns
        -------
        p: float
            sequentially valid p-value of the hypothesis that the population mean is less than or equal to t
        p_history: numpy array
            sample by sample history of p-values. Not meaningful unless the sample is in random order.
        '''
        u = kwargs.get('u', 1)
        eta = kwargs.get('eta', u*(1-np.finfo(float).eps))
        if not estim:
            estim = TestNonnegMean.fixed_alternative_mean
        S = np.insert(np.cumsum(x),0,0)        # 0, x_1, x_1+x_2, ...,
        Stot = S[-1]                           # sample total
        S = S[0:-1]                            # same length as the data
        j = np.arange(1,len(x)+1)              # 1, 2, 3, ..., len(x)
        m = (N*t-S)/(N-j+1) if np.isfinite(N) else t   # mean of population after (j-1)st draw, if null is true
        etaj = estim(x=x, N=N, t=m, **kwargs)
        x = np.array(x)
        with np.errstate(divide='ignore',invalid='ignore'):
            terms = np.cumprod((x*etaj/m + (u-x)*(u-etaj)/(u-m))/u)
        terms[m>u] = 0                                              # true mean is certainly less than hypothesized
        terms[np.isclose(0, m, atol=2*np.finfo(float).eps)] = 1     # ignore
        terms[np.isclose(u, m, atol=10**-8, rtol=10**-6)] = 1       # ignore
        terms[np.isclose(0, terms, atol=2*np.finfo(float).eps)] = 1 # martingale effectively vanishes; p-value 1
        terms[m<0] = np.inf                                         # true mean certainly greater than hypothesized
        terms[-1] = (np.inf if Stot > N*t else terms[-1])           # final sample maked the total greater than the null
        return min(1, 1/np.max(terms)), np.minimum(1,1/terms)


    @classmethod
    def kaplan_markov(cls, x: np.array, t: float=1/2, **kwargs) -> typing.Tuple[float, np.array]:
        '''
        Kaplan-Markov p-value for the hypothesis that the sample x is drawn IID from a population
        with mean t against the alternative that the mean is less than t.

        If there is a possibility that x has elements equal to zero, set g>0; otherwise, the p-value
        will be 1.

        If the order of the values in the sample is random, you can set random_order = True to use
        optional stopping to increase the power. If the values are not in random order or if you want
        to use all the data, set random_order = False

        Parameters:
        -----------
        x: array-like
            the sample
        t: float
            the null value of the mean
        kwargs:
            g: float
                "padding" so that if there are any zeros in the sample, the martingale doesn't vanish forever
            random_order: Boolean
                if the sample is in random order, it is legitimate to stop early, which can yield a 
                more powerful test. See above.

        Returns:
        --------
        p: float
           the p-value
        p_history: numpy array
            sample by sample history of p-values. Not meaningful unless the sample is in random order.
        '''
        g = kwargs.get('g', 0)
        random_order = kwargs.get('random_order', True)
        if any(xx < 0 for xx in x):
            raise ValueError('Negative value in sample from a nonnegative population.')
        p_history = np.cumprod((t+g)/(x+g))
        return np.min([1, np.min(p_history) if random_order else p_history[-1]]), \
               np.minimum(p_history,1)

    @classmethod
    def kaplan_wald(cls, x: np.array, t: float=1/2, **kwargs) -> typing.Tuple[float, np.array]:
        '''
        Kaplan-Wald p-value for the hypothesis that the sample x is drawn IID from a population
        with mean t against the alternative that the mean is less than t.

        If there is a possibility that x has elements equal to zero, set g \in (0, 1);
        otherwise, the p-value will be 1.

        If the order of the values in the sample is random, you can set random_order = True to use
        optional stopping to increase the power. If the values are not in random order or if you want
        to use all the data, set random_order = False

        Parameters:
        -----------
        x: array-like
            the sample
        t: float
            the null value of the mean
        kwargs:
            g: float
                "padding" in case there any values in the sample are zero
            random_order: Boolean
                if the sample is in random order, it is legitimate to stop early, which
                can yield a more powerful test. See above.

        Returns:
        --------
        p: float
            p-value
        p_history: numpy array
            sample by sample history of p-values. Not meaningful unless the sample is in random order.

        '''
        g = kwargs.get('g', 0)
        random_order = kwargs.get('random_order', True)
        if g < 0:
            raise ValueError('g cannot be negative')
        if any(xx < 0 for xx in x):
            raise ValueError('Negative value in sample from a nonnegative population.')
        p_history = np.cumprod((1-g)*x/t + g)
        return np.min([1, 1/np.max(p_history) if random_order \
                       else 1/p_history[-1]]), np.minimum(1/p_history, 1)

    @classmethod
    def kaplan_kolmogorov(cls, x: np.array, N: int, t: float=1/2, **kwargs) -> typing.Tuple[float, np.array]:
        '''
        p-value for the hypothesis that the mean of a nonnegative population with N
        elements is t. The alternative is that the mean is less than t.
        If the random sample x is in the order in which the sample was drawn, it is
        legitimate to set random_order = True.
        If not, set random_order = False.

        g is a tuning parameter to protect against data values equal to zero.
        g should be in [0, 1)

        Parameters
        ----------
        x: list
            observations
        N: int
            population size
        t: float
            null value of the population mean
        kwargs:
            g: float
                "padding" in case there any values in the sample are zero
            random_order: Boolean
                if the sample is in random order, it is legitimate to stop early, which can yield a 
                more powerful test. See above.

        Returns
        -------
        p: float
           the p-value
        p_history: numpy array
            sample by sample history of p-values. Not meaningful unless the sample is in random order.
        '''
        g = kwargs.get('g', 0)
        random_order = kwargs.get('random_order', True)
        x = np.array(x)
        assert all(x >=0),  'Negative value in a nonnegative population!'
        assert len(x) <= N, 'Sample size is larger than the population!'
        assert N > 0,       'Population size not positive!'
        assert N == int(N), 'Non-integer population size!'

        S = np.insert(np.cumsum(x+g),0,0)[0:-1]  # 0, x_1, x_1+x_2, ...,
        j = np.arange(1,len(x)+1)              # 1, 2, 3, ..., len(x)
        m = (N*(t+g)-S)/(N-j+1) if np.isfinite(N) else t+g   # mean of population after (j-1)st draw, if null is true
        with np.errstate(divide='ignore',invalid='ignore'):
            terms = np.cumprod((x+g)/m)
        terms[m<0] = np.inf
        p = min((1/np.max(terms) if random_order else 1/terms[-1]),1)
        return p, np.minimum(1/terms, 1)
    
    @classmethod
    def sample_size(
                    cls, risk_fn: callable, x: np.array, N: int, t: float=1/2, alpha: float=0.05, 
                    reps: int=None, prefix: bool=False, quantile: float=0.5, **kwargs) -> int:
        '''
        Estimate the sample size to reject the null hypothesis that the population mean of a population of size 
        `N` is `<=t` at significance level `alpha`, using pilot data `x`.

        If `reps is None`, concatenates copies of `x` to produce a list of length `N`.

        If `reps is not None`, samples at random from `x` to produce `reps` lists of length `N` and reports 
        the `quantile` quantile of the resulting sample sizes.

        If `prefix == True`, starts every list with `x` as given, then samples from `x` to produce the 
        remaining `N-len(x)` entries.

        Parameters
        ----------
        risk_fn: callable
            the risk function, with calling signature risk_fn(x, N, t, **kwargs)
            It should return a list of p-values, corresponding to observing x sequentially.
        x: list or np.array
            the data for the simulation or calculation
        N: int
            the size of the entire population
        t: nonnegative float 
            the hypothesized population mean
        alpha: float in (0, 1/2)
            the significance level for the test
        reps: int
            the number of replications to use to simulate the sample size. If `reps is None`, estimates deterministically
        prefix: bool
            whether to use `x` as the prefix of the data sequence. Not relevant if `reps is None`
        quantile: float in (0, 1)
            desired quantile of the sample sizes from simulations. Not used if `reps is None`
        kwargs:
            arguments passed to risk_fn()
        '''
        if reps is None:
            pop = np.repeat(np.array(x), math.ceil(N/len(x)))[0:N]  # replicate data to make the population
            p = risk_fn(pop, N, t, **kwargs)
            crossed = (p<=alpha)
            sam_size = int(N if np.sum(crossed)==0 else (np.argmax(crossed)+1))
        else:  # estimate the quantile by simulation
            seed = kwargs.get('seed',1234567890)
            prng = np.random.RandomState(seed)  # use the Mersenne Twister for speed
            sams = np.zeros(int(reps))
            pfx = x if prefix else []
            ran_len = (N-len(x)) if prefix else N
            for r in range(reps):
                pop = np.append(pfx, prng.choice(x, size=ran_len, replace=True))
                p = risk_fn(pop, N, t, **kwargs)
                crossed = (p<=alpha)
                sams[r] = N if np.sum(crossed)==0 else (np.argmax(crossed)+1)
            sam_size = int(np.quantile(sams, quantile))
        return sam_size

                     

### Unit tests

def test_alpha_mart():
    eps = 0.0001  # Generic small value

    # When all the items are 1/2, estimated p for a mean of 1/2 should be 1.
    s = np.ones(5)/2
    np.testing.assert_almost_equal(TestNonnegMean.alpha_mart(s, N=100000, t=1/2)[0],1.0)
    np.testing.assert_array_less(TestNonnegMean.alpha_mart(s, N=100000, t=eps)[:1],[eps])

    s = [0.6,0.8,1.0,1.2,1.4]
    np.testing.assert_array_less(TestNonnegMean.alpha_mart(s, N=100000, t=eps)[:1],[eps])

    s1 = [1, 0, 1, 1, 0, 0, 1]
    alpha_mart1 = TestNonnegMean.alpha_mart(s1, N=7, t=3/7)[1]
    # p-values should be big until the last, which should be 0
    print(f'{alpha_mart1=}')
    assert(not any(np.isnan(alpha_mart1)))
    assert(alpha_mart1[-1] == 0)

    s2 = [1, 0, 1, 1, 0, 0, 0]
    alpha_mart2 = TestNonnegMean.alpha_mart(s2, N=7, t=3/7)[1]
    # Since s1 and s2 only differ in the last observation,
    # the resulting martingales should be identical up to the next-to-last.
    # Final entry in alpha_mart2 should be 1
    assert(all(np.equal(alpha_mart2[0:(len(alpha_mart2)-1)],
                        alpha_mart1[0:(len(alpha_mart1)-1)])))
    print(f'{alpha_mart2=}')

def test_shrink_trunc():
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
    for eta in etas:
        c = (eta-t)/2
        for x in v:
            N = len(x)
            xinf = TestNonnegMean.shrink_trunc(x, N=np.inf, t=t, u=u, eta=eta, c=c, d=d, f=f)
            xfin = TestNonnegMean.shrink_trunc(x, N=N,      t=t, u=u, eta=eta, c=c, d=d, f=f)
            yinf = np.zeros(N)
            yfin = np.zeros(N)
            for j in range(1,N+1):
                est = (d*eta + Sj(x,j))/(d+j-1)
                most = u*(1-np.finfo(float).eps)
                yinf[j-1] = np.minimum(np.maximum(t+epsj(c,d,j), est), most)
                yfin[j-1] = np.minimum(np.maximum(tj(N,t,x,j)+epsj(c,d,j), est), most)
            np.testing.assert_allclose(xinf, yinf)
            np.testing.assert_allclose(xfin, yfin)

def test_kaplan_markov():
    s = np.ones(5)
    np.testing.assert_almost_equal(TestNonnegMean.kaplan_markov(s)[0], 2**-5)
    s = np.array([1, 1, 1, 1, 1, 0])
    np.testing.assert_almost_equal(TestNonnegMean.kaplan_markov(s, g=0.1)[0],(1.1/.6)**-5)
    np.testing.assert_almost_equal(TestNonnegMean.kaplan_markov(s, g=0.1, random_order = False)[0],(1.1/.6)**-5 * .6/.1)
    s = np.array([1, -1])
    try:
        TestNonnegMean.kaplan_markov(s)
    except ValueError:
        pass
    else:
        raise AssertionError

def test_kaplan_wald():
    s = np.ones(5)
    np.testing.assert_almost_equal(TestNonnegMean.kaplan_wald(s)[0], 2**-5)
    s = np.array([1, 1, 1, 1, 1, 0])
    np.testing.assert_almost_equal(TestNonnegMean.kaplan_wald(s, g=0.1)[0], (1.9)**-5)
    np.testing.assert_almost_equal(TestNonnegMean.kaplan_wald(s, g=0.1, random_order = False)[0],\
                                   (1.9)**-5 * 10)
    s = np.array([1, -1])
    try:
        TestNonnegMean.kaplan_wald(s)
    except ValueError:
        pass
    else:
        raise AssertionError

def test_kaplan_kolmogorov():
    # NEEDS WORK! This just prints; it doesn't really test anything.
    N = 100
    x = np.ones(10)
    p1 = TestNonnegMean.kaplan_kolmogorov(x, N, t=1/2, g=0, random_order = True)
    x = np.zeros(10)
    p2 = TestNonnegMean.kaplan_kolmogorov(x, N, t=1/2, g=0.1, random_order = True)
    print("kaplan_kolmogorov: {} {}".format(p1, p2))
    
def test_sample_size():
    estim = TestNonnegMean.fixed_alternative_mean
    eta = 0.75
    u=1
    N = 1000
    x = np.ones(math.floor(N/200))
    t = 1/2
    alpha = 0.05
    prefix = True
    quantile = 0.5
    reps=None
    prefix=False
    risk_fn = lambda x, N, t, **kwargs: TestNonnegMean.alpha_mart(x, N, t, estim=estim, eta=eta, u=u)[1]
    sam_size = TestNonnegMean.sample_size(risk_fn=risk_fn, x=x, N=N, t=t, alpha=alpha, reps=reps, prefix=prefix, quantile=quantile)
    sam_size = TestNonnegMean.sample_size(risk_fn=risk_fn, x=x, N=N, t=t, alpha=alpha, reps=reps, prefix=prefix, quantile=quantile)
    np.testing.assert_equal(sam_size, 8) # ((.75/.5)*1+(.25/.5)*0)**8 = 25 > 1/alpha, so sam_size=8
#    
    reps=100
    sam_size = TestNonnegMean.sample_size(risk_fn=risk_fn, x=x, N=N, t=t, alpha=alpha, reps=reps, prefix=prefix, quantile=quantile)
    np.testing.assert_equal(sam_size, 8) # all simulations should give the same answer
#
    x = 0.75*np.ones(math.floor(N/200))
    sam_size = TestNonnegMean.sample_size(risk_fn=risk_fn, x=x, N=N, t=t, alpha=alpha, reps=reps, prefix=prefix, quantile=quantile)
    np.testing.assert_equal(sam_size, 14) # ((.75/.5)*.75+(.25/.5)*.25)**14 = 22.7 > 1/alpha, so sam_size=14    
    
    
#####
if __name__ == "__main__":
    test_kaplan_markov()
    test_kaplan_wald()
    test_kaplan_kolmogorov()
    test_alpha_mart()
    test_shrink_trunc()
    test_sample_size()
