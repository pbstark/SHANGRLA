import numpy as np

# Make sure shangrla is in your PYTHONPATH
from shangrla.core.NonnegMean import NonnegMean
from shangrla.core.Audit import Assertion


def sample_size(mean, tw, tl, to, args, N, upper_bound=1, polling=False):

    margin = 2*mean - 1
    u = 2/(2-(margin/upper_bound))

    test = NonnegMean(test=NonnegMean.alpha_mart, \
        estim=NonnegMean.shrink_trunc, N=N, u=u, eta=mean) if \
        polling else NonnegMean(test=NonnegMean.alpha_mart, \
        estim=NonnegMean.optimal_comparison, N=N, u=u, eta=mean)


    # over: (1 - o/u)/(2 - v/u)

    # where o is the overstatement, u is the upper bound on the value
    # assorter assigns to any ballot, v is the assorter margin.
    big = 1 if polling else (1 / (2 - margin/upper_bound)) # o=0
    small = 0 if polling else (0.5 / (2 - margin/upper_bound)) # o=0.5

    r1 = args.erate1
    r2 = args.erate2

    x = big*np.ones(N)

    if polling:
        n_0 = tl
        n_big = tw
        n_half = to

        x = Assertion.interleave_values(n_0, n_half, n_big, big=big)
    else:
        rate_1_i = np.arange(0, N, step=int(1/r1), dtype=int) if r1 else []
        rate_2_i = np.arange(0, N, step=int(1/r2), dtype=int) if r2 else []

        x[rate_1_i] = small
        x[rate_2_i] = 0

    return test.sample_size(x, alpha=args.rlimit, reps=args.reps, \
        seed=args.seed, random_order=True)


def bp_estimate(winner, loser, other, total):
    p = (winner+loser)/total
    q = (winner-loser)/(winner+loser)

    margin = p * (q*q)

    return 1.0/margin


def cp_estimate(winner, loser, other, total):
    amargin = 2*((winner+0.5*other)/total) - 1

    return 1.0/amargin


