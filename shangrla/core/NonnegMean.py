import math
import numpy as np
import warnings

##########################################################################################


def welford_mean_var(x: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """
    Welford's algorithm for running mean and variance
    """
    m = [x[0]]
    v = [0]
    for i, xi in enumerate(x[1:]):
        m.append(m[-1] + (xi - m[-1]) / (i + 2))
        v.append(v[-1] + (xi - m[-2]) * (xi - m[-1]))
    v = v / np.arange(1, len(x) + 1)
    return np.array(m), v


class NonnegMean:
    """
    Tests of the hypothesis that the mean of a population of values in [0, u] is less than or equal to t.
        Several tests are implemented, all ultimately based on martingales or supermartingales:
            Kaplan-Kolmogorov (with and without replacement)
            Kaplan-Markov (without replacement)
            Kaplan-Wald (without replacement)
            Wald SPRT (with and with replacement)
            ALPHA supermartingale test (with and without replacement)
            Betting martingale tests (with and without replacement)
    Some tests work for all nonnegative populations; others require a finite upper bound `u`.
    Many of the tests have versions for sampling with replacement (`N=np.inf`) and for sampling
    without replacement (`N` finite).
    Betting martingales and ALPHA martingales are different parametrizations of the same tests, but
      lead to different heuristics for selecting the parameters.
    """

    TESTS = (
        ALPHA_MART := "ALPHA_MART",
        BETTING_MART := "BETTING_MART",
        KAPLAN_KOLMOGOROV := "KAPLAN_KOLMOGOROV",
        KAPLAN_MARKOV := "KAPLAN_MARKOV",
        KAPLAN_WALD := "KAPLAN_WALD",
        WALD_SPRT := "WALD_SPRT",
    )

    def __init__(
        self,
        test: callable = None,
        estim: callable = None,
        bet: callable = None,
        u: float = 1,
        N: int = np.inf,
        t: float = 1 / 2,
        random_order: bool = True,
        **kwargs,
    ):
        """
        kwargs can be used to set attributes such as `betting` and parameters later used by
        `test`, `estim`, or `bet`, for instance, `eta` and to pass `c`, `d`, `f`, and `minsd` to
        `shrink_trunc()` or other estimators or betting strategies.
        """
        if test is None:  # default to alpha_mart
            test = self.alpha_mart
        if estim is None:
            estim = self.fixed_alternative_mean
            self.eta = kwargs.get(
                "eta", t + (u - t) / 2
            )  # initial estimate of population mean
        if bet is None:
            bet = self.fixed_bet
            self.lam = kwargs.get("lam", 0.5)  # initial fraction of fortune to bet
        self.test = test.__get__(self)
        self.estim = estim.__get__(self)
        self.bet = bet.__get__(self)
        self.u = u
        self.N = N
        self.t = t
        self.random_order = random_order
        self.kwargs = kwargs  # preserving these for __str__()
        self.__dict__.update(kwargs)

    def __str__(self):
        return (
            f"test: {self.test} estim: {self.estim} upper bound u: {self.u} "
            f"N: {self.N} null mean t: {self.t} kwargs: {self.kwargs}"
        )

    def alpha_mart(self, x: np.array, **kwargs) -> tuple[float, np.array]:
        """
        Finds the ALPHA martingale for the hypothesis that the population
        mean is less than or equal to t using a martingale method,
        for a population of size N, based on a series of draws x.

        **The draws must be in random order**, or the sequence is not a supermartingale under the null

        If N is finite, assumes the sample is drawn without replacement
        If N is infinite, assumes the sample is with replacement

        Parameters
        ----------
        x: list corresponding to the data
        attributes used:
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
        """
        N = self.N
        t = self.t
        u = self.u
        atol = kwargs.get("atol", 2 * np.finfo(float).eps)
        rtol = kwargs.get("rtol", 10**-6)
        _S, Stot, _j, m = self.sjm(N, t, x)
        x = np.array(x)
        with np.errstate(divide="ignore", invalid="ignore", over="ignore"):
            etaj = self.estim(x)
            terms = np.cumprod((x * etaj / m + (u - x) * (u - etaj) / (u - m)) / u)
        terms[m > u] = 0  # true mean is certainly less than hypothesized
        terms[np.isclose(0, m, atol=atol)] = 1  # ignore
        terms[np.isclose(u, m, atol=atol, rtol=rtol)] = 1  # ignore
        terms[np.isclose(0, terms, atol=atol)] = (
            1  # martingale effectively vanishes; p-value 1
        )
        terms[m < 0] = np.inf  # true mean certainly greater than hypothesized
        terms[-1] = (
            np.inf if Stot > N * t else terms[-1]
        )  # final sample makes the total greater than the null
        return min(1, 1 / np.max(terms)), np.minimum(1, 1 / terms)

    def sjm(self, N: int, t: float, x: np.array) -> tuple[np.array, float, np.array, np.array]:
        """
        This method calculates the cumulative sum of the input array `x`, the total sum of `x`,
        an array of indices, and the mean of the population after each draw if the null hypothesis is true.

        Parameters
        ----------
        N : int or float
            The size of the population. If N is np.inf, it means the sampling is with replacement.
        t : float
            The hypothesized population mean under the null hypothesis.
        x : np.array
            The input data array.

        Returns
        -------
        S : np.array
            The cumulative sum of the input array `x`, excluding the last element.
        Stot : float
            The total sum of the input array `x`.
        j : np.array
            An array of indices from 1 to the length of `x`.
        m : np.array
            The mean of the population after each draw if the null hypothesis is true.
        """
        assert isinstance(N, int) or (math.isinf(N) and N > 0), "Population size is not an integer!"
        S = np.insert(np.cumsum(x), 0, 0)  # 0, x_1, x_1+x_2, ...,
        Stot = S[-1]  # sample total
        S = S[0:-1]  # same length as the data
        j = np.arange(1, len(x) + 1)  # 1, 2, 3, ..., len(x)
        assert j[-1] <= N, "Sample size is larger than the population!"
        m = (
            (N * t - S) / (N - j + 1) if np.isfinite(N) else t
        )  # mean of population after (j-1)st draw, if null is true (t=eta is the mean)
        return S, Stot, j, m

    def betting_mart(self, x: np.array, **kwargs) -> tuple[float, np.array]:
        """
        Finds the betting martingale for the hypothesis that the population
        mean is less than or equal to t using a martingale method,
        for a population of size N, based on a series of draws x.

        **The draws must be in random order**, or the sequence is not a supermartingale under the null

        If N is finite, assumes the sample is drawn without replacement
        If N is infinite, assumes the sample is with replacement

        Parameters
        ----------
        x: list corresponding to the data
        attributes used:
            keyword arguments for bet() and for this function
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
        """
        N = self.N
        t = self.t
        u = self.u
        atol = kwargs.get("atol", 2 * np.finfo(float).eps)
        rtol = kwargs.get("rtol", 10**-6)
        _S, Stot, _j, m = self.sjm(N, t, x)
        x = np.array(x)
        with np.errstate(divide="ignore", invalid="ignore", over="ignore"):
            lam = self.bet(x)
            terms = np.cumprod(1 + lam * (x - m))
        terms[m > u] = 0  # true mean is certainly less than hypothesized
        terms[np.isclose(0, m, atol=atol)] = 1  # ignore
        terms[np.isclose(u, m, atol=atol, rtol=rtol)] = 1  # ignore
        terms[np.isclose(0, terms, atol=atol)] = (
            1  # martingale effectively vanishes; p-value 1
        )
        terms[m < 0] = np.inf  # true mean certainly greater than hypothesized
        terms[-1] = (
            np.inf if Stot > N * t else terms[-1]
        )  # final sample makes the total greater than the null
        return min(1, 1 / np.max(terms)), np.minimum(1, 1 / terms)

    def fixed_alternative_mean(self, x: np.array, **kwargs) -> np.array:
        """
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
        kwargs:
            eta: float in (t, u) (default u*(1-eps))
                alternative hypothethesized value for the population mean
            u: float > 0 (default 1)
                upper bound on the population values
        """
        u = self.u
        N = self.N
        eta = getattr(self, "eta", u * (1 - np.finfo(float).eps))
        _S, _Stot, _j, m = self.sjm(N, eta, x)
        if (negs := np.sum(m < 0)) > 0:
            warnings.warn(
                f"Implied population mean is negative in {negs} of {len(x)} terms"
            )
        if (pos := np.sum(m > u)) > 0:
            warnings.warn(
                f"Implied population mean is greater than {u=} in {pos} of {len(x)} terms"
            )
        return m

    def shrink_trunc(self, x: np.array, **kwargs) -> np.array:
        """
        apply shrinkage/truncation estimator to an array to construct a sequence of "alternative" values

        sample mean is shrunk towards eta, with relative weight d compared to a single observation,
        then that combination is shrunk towards u, with relative weight f/(stdev(x)).

        The result is truncated above at u*(1-eps) and below at m_j+e_j(c,j)

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
        attributes used:
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
        """
        # set the parameters
        u = self.u
        N = self.N
        t = self.t
        eta = getattr(self, "eta", u * (1 - np.finfo(float).eps))
        c = getattr(self, "c", 1 / 2)
        d = getattr(self, "d", 100)
        f = getattr(self, "f", 0)
        minsd = getattr(self, "minsd", 10**-6)
        S, _, j, m = self.sjm(N, t, x)
        _, v = welford_mean_var(x)
        sdj = np.sqrt(v)
        # threshold the sd, set first two sds to 1
        sdj = np.insert(np.maximum(sdj, minsd), 0, 1)[0:-1]
        sdj[1] = 1
        weighted = ((d * eta + S) / (d + j - 1) + u * f / sdj) / (1 + f / sdj)
        return np.minimum(
            u * (1 - np.finfo(float).eps),
            np.maximum(weighted, m + c / np.sqrt(d + j - 1)),
        )

    def optimal_comparison(self, x: np.array, **kwargs) -> np.array:
        """
        The value of eta corresponding to the "bet" that is optimal for ballot-level comparison audits,
        for which overstatement assorters take a small number of possible values and are concentrated
        on a single value when the CVRs have no errors.

        Let p0 be the rate of error-free CVRs, p1=0 the rate of 1-vote overstatements,
        and p2= 1-p0-p1 = 1-p0 the rate of 2-vote overstatements. Then

        eta = (1-u*p0)/(2-2*u) + u*p0 - 1/2, where p0 is the rate of error-free CVRs.

        Translating to p2=1-p0 gives:

        eta = (1-u*(1-p2))/(2-2*u) + u*(1-p2) - 1/2.

        Parameters
        ----------
        x: np.array
            input data
        rate_error_2: float
            hypothesized rate of two-vote overstatements

        Returns
        -------
        eta: float
            estimated alternative mean to use in alpha
        """
        # set the parameters
        # TO DO: double check where rate_error_2 is set
        p2 = getattr(self, "rate_error_2", 1e-4)  # rate of 2-vote overstatement errors
        return (1 - self.u * (1 - p2)) / (2 - 2 * self.u) + self.u * (1 - p2) - 1 / 2

    def fixed_bet(self, x: np.array, **kwargs) -> np.array:
        """
        Return a fixed value of lambda, the fraction of the current fortune to bet.

        Parameters
        ----------
        x: np.array
            input data

        Assumes the instance variable `lam` has been set.
        """
        return self.lam * np.ones_like(x)

    def agrapa(self, x: np.array, **kwargs) -> np.array:
        """
        maximize approximate growth rate adapted to the particular alternative (aGRAPA) bet of Waudby-Smith & Ramdas (WSR)

        This implementation alters the method from support \mu \in [0, 1] to \mu \in [0, u], and to constrain
        the bets to be positive (for one-sided tests against the alternative that the true mean is larger than
        hypothesized)

        lam_j := 0 \vee (\hat{\mu}_{j-1}-t)/(\hat{\sigma}_{j-1}^2 + (t-\hat{\mu})^2) \wedge c_grapa/t

        \hat{\sigma} is the standard deviation of the sample.

        \hat{\mu} is the mean of the sample

        The value of c_grapa \in (0, 1) is passed as an instance variable of Class NonnegMean

        The running standard deviation is calculated using Welford's method.

        S_0 := 0
        S_j := \sum_{i=0}^{j-1} x_i, j >= 1
        t_adj := (N*t-S_j)/(N-j+1) if np.isfinite(N) else t
        sd_0 := 0
        sd_j := sqrt[(\sum_{i=1}^{j-1} (x_i-S_j/(j-1))^2)/(j-2)] \wedge minsd, j>2
        lam_1 := self.lam
        lam_j :=  0 \vee (\hat{m_{j-1}-t)/(sd_{j-1}^2 + (t-m_{j-1})^2) \wedge c_grapa/t

        Parameters
        ----------
        x: np.array
            input data
        attributes used:
            c_grapa_0: float in (0, 1)
                initial scale factor c_j in WSR's agrapa bet
            c_grapa_max: float in (1, 1-np.finfo(float).eps]
                asymptotic limit of the value of c_j
            c_grapa_grow: float in [0, np.infty)
                rate at which to allow c to grow towards c_grapa_max.
                c_j := c_grapa_0 + (c_grapa_max-c_grapa_0)*(1-1/(1+c_grapa_grow*np.sqrt(j)))
                A value of 0 keeps c_j equal to c_grapa for all j.

        """
        # set the parameters
        u = self.u  # population upper bound
        N = self.N  # population size
        t = self.t  # hypothesized population mean
        lam = getattr(self, "lam", 0.5)  # initial bet
        c_g_0 = getattr(
            self, "c_grapa_0", (1 - np.finfo(float).eps)
        )  # initial truncation value c for agrapa
        c_g_m = getattr(
            self, "c_grapa_max", (1 - np.finfo(float).eps)
        )  # asymptotic limit of c
        c_g_g = getattr(self, "c_grapa_grow", 0)  # rate to let c grow towards c_g_m
        mj, sdj2 = welford_mean_var(x)
        t_adj = (
            (N * t - np.insert(np.cumsum(x), 0, 0)[0:-1]) / (N - np.arange(len(x)))
            if np.isfinite(N)
            else t * np.ones(len(x))
        )
        lamj = (mj - t_adj) / (sdj2 + (t_adj - mj) ** 2)  # agrapa bet
        #  shift and set first bet to self.lam
        lamj = np.insert(lamj, 0, lam)[0:-1]
        c = c_g_0 + (c_g_m - c_g_0) * (1 - 1 / (1 + c_g_g * np.sqrt(np.arange(len(x)))))
        lamj = np.maximum(0, np.minimum(c / t_adj, lamj))
        return lamj

    def lam_to_eta(self, lam: np.array, mu: np.array) -> np.array:
        """
        Convert bets (lam) for betting martingale to their implied estimates of the mean, eta, for ALPHA

        Parameters
        ----------
        lam: float or numpy array
            the value(s) of lam (the fraction of the current fortune to bet on the next draw)
        mu: float or numpy array
            sequence of population mean(s) if the null is true, adjusted for values already seen

        Returns
        -------
        eta: float or numpy array
            the corresponding value(s) of the mean
        """
        return mu * (1 + lam * (self.u - mu))

    def eta_to_lam(self, eta: np.array, mu: np.array) -> np.array:
        """
        Convert eta for ALPHA to corresponding bet lam for the betting martingale parametrization

        Parameters
        ----------
        eta: float or numpy array
            the value(s) of lam (the fraction of the current fortune to bet on the next draw)
        mu: float or numpy array
            sequence of population mean(s) if the null is true, adjusted for values already seen

        Returns
        -------
        lam: float or numpy array
            the corresponding betting fractions
        """
        return (eta / mu - 1) / (self.u - mu)

    def kaplan_kolmogorov(self, x: np.array, **kwargs) -> tuple[float, np.array]:
        """
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
        attributes used:
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
        """
        N = self.N
        t = self.t
        g = getattr(self, "g", 0)
        random_order = getattr(self, "random_order", True)
        x = np.array(x)
        assert all(x >= 0), "Negative value in a nonnegative population!"
        assert len(x) <= N, "Sample size is larger than the population!"
        assert N > 0, "Population size not positive!"
        assert N == int(N), "Non-integer population size!"

        _S, _Stot, _j, m = self.sjm(N, t+g, x+g)
        with np.errstate(divide="ignore", invalid="ignore", over="ignore"):
            terms = np.cumprod((x + g) / m)
        terms[m < 0] = np.inf
        p = min((1 / np.max(terms) if random_order else 1 / terms[-1]), 1)
        return p, np.minimum(1 / terms, 1)

    def kaplan_markov(self, x: np.array, **kwargs) -> tuple[float, np.array]:
        """
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
        attributes used:
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
        """
        t = self.t
        g = getattr(self, "g", 0)
        random_order = getattr(self, "random_order", True)
        if negs := sum(xx < 0 for xx in x) > 0:
            raise ValueError(
                "{negs} negative values in sample from a nonnegative population."
            )
        p_history = np.cumprod((t + g) / (x + g))
        return np.min(
            [1, np.min(p_history) if random_order else p_history[-1]]
        ), np.minimum(p_history, 1)

    def kaplan_wald(self, x: np.array, **kwargs) -> tuple[float, np.array]:
        """
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
        attributes used:
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

        """
        g = getattr(self, "g", 0)
        random_order = getattr(self, "random_order", True)
        t = self.t
        if g < 0 or g > 1:
            raise ValueError(f"{g=}, but g must be between 0 and 1. ")
        if any(xx < 0 for xx in x):
            raise ValueError("Negative value in sample from a nonnegative population.")
        p_history = np.cumprod((1 - g) * x / t + g)
        return np.min(
            [1, 1 / np.max(p_history) if random_order else 1 / p_history[-1]]
        ), np.minimum(1 / p_history, 1)

    def wald_sprt(self, x: np.array, **kwargs) -> tuple[float, np.array]:
        """
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
        x: list
            values between 0 and u
        kwargs:
            eta: float in (0,u)
                alternative hypothesized population mean
            random_order: Boolean
                if the data are in random order, setting this to True can improve the power.
                If the data are not in random order, set to False

        Returns
        -------
        tuple:
            p: float
                p-value
            p_history: numpy array
                sample by sample history of p-values. Not meaningful unless the sample is in random order.
        """
        u = self.u
        N = self.N
        t = self.t
        eta = getattr(self, "eta", u * (1 - np.finfo(float).eps))
        random_order = getattr(self, "random_order", True)
        if any((xx < 0 or xx > u) for xx in x):
            raise ValueError(f"Data out of range [0,{u}]")
        if np.isfinite(N):
            if not random_order:
                raise ValueError(
                    "data must be in random order for samples without replacement"
                )
            S = np.insert(np.cumsum(x), 0, 0)[0:-1]  # 0, x_1, x_1+x_2, ...,
            j = np.arange(1, len(x) + 1)  # 1, 2, 3, ..., len(x)
            m = (N * t - S) / (
                N - j + 1
            )  # mean of population after (j-1)st draw, if null is true
            etas = (N * eta - S) / (N - j + 1)
        else:
            m = t
            etas = eta
        with np.errstate(divide="ignore", invalid="ignore", over="ignore"):
            terms = np.cumprod(
                (x * etas / m + (u - x) * (u - etas) / (u - m)) / u
            )  # generalization of Bernoulli SPRT
        terms[m < 0] = np.inf  # the null is surely false
        terms = np.cumprod(terms)
        return 1 / np.max(terms) if random_order else 1 / terms[-1], np.minimum(
            1, 1 / terms
        )

    def sample_size(self, x: list = None, alpha: float = 0.05, reps: int = None, prefix: bool = False,
                    quantile: float = 0.5, **kwargs) -> int:
        """
        Estimate the sample size to reject the null hypothesis that the population mean of a population of size
        `N` is `<=t` at significance level `alpha`, using pilot data `x`.

        If `reps is None`, tiles copies of `x` to produce a list of length `N`.

        If `reps is not None`, samples at random from `x` to produce `reps` lists of length `N` and reports
        the `quantile` quantile of the resulting sample sizes.

        If `prefix == True`, starts every list with `x` as given, then samples from `x` to produce the
        remaining `N-len(x)` entries.

        Parameters
        ----------
        x: list or np.array
            the data for the simulation or calculation
        alpha: float in (0, 1/2)
            the significance level for the test
        reps: int
            the number of replications to use to simulate the sample size. If `reps is None`, estimates deterministically
        prefix: bool
            whether to use `x` as the prefix of the data sequence. Not used if `reps is None`
        quantile: float in (0, 1)
            desired quantile of the sample sizes from simulations. Not used if `reps is None`
        kwargs:
            keyword args passed to self.test()

        Returns
        -------
        sam_size: int
            estimated sample size
        """
        N = self.N
        if reps is None:
            pop = np.repeat(np.array(x), math.ceil(N / len(x)))[
                0:N
            ]  # tile data to make the population
            p = self.test(pop, **kwargs)[1]
            crossed = p <= alpha
            sam_size = int(N if np.sum(crossed) == 0 else (np.argmax(crossed) + 1))
        else:  # estimate the quantile by a bootstrap-like simulation
            seed = kwargs.get("seed", 1234567890)
            prng = np.random.RandomState(seed)  # use the Mersenne Twister for speed
            sams = np.zeros(int(reps))
            pfx = np.array(x) if prefix else []
            ran_len = (N - len(x)) if prefix else N
            for r in range(reps):
                pop = np.append(pfx, prng.choice(x, size=ran_len, replace=True))
                p = self.test(pop, **kwargs)[1]
                crossed = p <= alpha
                sams[r] = N if np.sum(crossed) == 0 else (np.argmax(crossed) + 1)
            sam_size = int(np.quantile(sams, quantile))
        return sam_size
