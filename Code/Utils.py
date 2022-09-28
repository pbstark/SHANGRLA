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
from CVR import CVR
from Assertion import Assertion, Assorter
from TestNonnegMean import TestNonnegMean


class Utils:
    '''
    Utilities for SHANGRLA RLAs
    '''
    
    @classmethod
    def assorter_initial_sample_size(
                            cls, risk_fn: callable, N: int, margin: float, t: float=1/2, u: float=1,
                            alpha: float=0.05, polling: bool=False, error_rate: float=0, reps: int=None,
                            bias_up: bool=True, quantile: float=0.5, seed: int=1234567890) -> int:
        '''
        Estimate sample size needed to reject the null hypothesis that a population mean is <=t at 
        significance level alpha, for the specified risk function, given the margin
        and--for comparison audits--assumptions about the rate of overstatement errors.

        If `polling == True`, bases estimates on the margin alone, for a ballot-polling audit using sampling 
        without replacement.

        If `polling == False`, uses `error_rate` as an assumed rate of one-vote overstatements.

        This function is for a single contest.

        Implements two strategies when `polling == False`:

            1. If reps is not None, uses simulations to estimate the `quantile` quantile
            of sample size required. The simulations use the numpy Mersenne Twister PRNG.

            2. If reps is None, puts discrepancies into the sample in a deterministic way, starting
            with a discrepancy in the first item if bias_up == True, then including an additional discrepancy after
            every int(1/error_rate) items in the sample. 
            "Frontloading" the errors (bias_up == True) should make this slightly conservative on average


        Parameters:
        -----------
        risk_function: callable
            risk function to use. risk_function should take four arguments (x, m, N, u) and return
            an array of p-values corresponding to the sample sequence x.
        N: int
            cards potentially containing the contest, or N = np.infty for sampling with replacement
        margin: float
            assorter margin. If the assorter is an overstatement assorter, it should be the overstatement assorter margin.
        t: float
            hypothesized value of the population mean
        u: float
            maximum value the assorter can assign to any ballot in the contest
            For example, 
                u=1 for polling audit of plurality contests
                u=2/(2-margin/original_assorter_upper_bound) for comparison audits
        alpha: float
            significance level in (0, 0.5)
        polling: bool
            True for polling audit, False for ballot-level comparison audit
        error_rate: float
            assumed rate of 1-vote overstatements for ballot-level comparison audit.
            Ignored if `polling==True`
        reps: int
            if reps is not none, performs `reps` simulations to estimate the `quantile` quantile of sample size
        bias_up: boolean
            if bias_up == True, front loads the discrepancies (biases sample size up).
            If bias_up == False, back loads the discrepancies (biases sample size down).
        quantile: float
            quantile of the distribution of sample sizes to return, if reps is not None.
            If reps is None, `quantile` is not used
        seed: int
            if reps is not None, use `seed` as the seed in numpy.random for simulations to estimate the quantile

        Returns:
        --------
        sample_size: int
            sample size estimated to be sufficient to confirm the outcome if one-vote overstatements
            in the sample are not more frequent than the assumed rate

        '''
        assert alpha > 0 and alpha < 1/2, f'{alpha=}. Not in (0, 1/2)'
        assert margin > 0, 'Margin is nonpositive'

        if not polling:          # ballot-level comparison audit
            clean = u/2          # 1/(2-margin/upper_bound) is the maximum possible value of the overstatement assorter
            one_vote_over = u/4  # (1-(upper_bound/2)/upper_bound)/(2-margin/upper_bound)
            if reps is None:     # allocate errors deterministically
                offset = 0 if bias_up else 1
                x = clean*np.ones(N)
                if error_rate > 0:
                    for k in range(N):
                        x[k] = (one_vote_over if (k+offset) % int(1/error_rate) == 0 else x[k])
                p = risk_fn(x, margin, N, u)
                crossed = (p<=alpha)
                sam_size = int(N if np.sum(crossed)==0 else (np.argmax(crossed)+1))
            else:                # estimate the quantile by simulation
                prng = np.random.RandomState(seed)  # use the Mersenne Twister for speed
                sams = np.zeros(int(reps))
                for r in range(reps):
                    pop = clean*np.ones(N)
                    inx = (prng.random(size=N) <= error_rate)  # randomly allocate errors
                    pop[inx] = one_vote_over
                    p = risk_fn(pop, margin, N, u)
                    crossed = (p<=alpha)
                    sams[r] = N if np.sum(crossed)==0 else (np.argmax(crossed)+1)
                sam_size = int(np.quantile(sams, quantile))
        else:                   # ballot-polling audit
            if reps is None:
                raise ValueError('estimating ballot-polling sample size requires setting `reps`')
            else: 
                pop = np.zeros(N)
                nonzero = math.floor((margin+1)/2)
                pop[0:nonzero] = np.ones(nonzero)
                prng = np.random.RandomState(seed)  # use the Mersenne Twister for speed
                sams = np.zeros(int(reps))
                for r in range(reps):
                    pop = prng.permutation(pop)
                    p = risk_function(pop, margin, N, u)
                    crossed = (p <= alpha)
                    sams[r] = N if np.sum(crossed)==0 else (np.argmax(crossed)+1)
                sam_size = int(np.quantile(sams, quantile))
        return sam_size

    @classmethod
    def check_audit_parameters(cls, risk_function, error_rate, contests):
        '''
        Check whether the audit parameters are valid; complain if not.

        Parameters:
        ---------
        risk_function: string
            the name of the risk-measuring function for the audit

        error_rate: float
            expected rate of 1-vote overstatements

        contests: dict of dicts
            contest-specific information for the audit

        Returns:
        --------
        '''
        assert error_rate >= 0, 'expected error rate must be nonnegative'
        for c in contests.keys():
            assert contests[c]['risk_limit'] > 0, 'risk limit must be nonnegative in ' + c + ' contest'
            assert contests[c]['risk_limit'] < 1, 'risk limit must be less than 1 in ' + c + ' contest'
            assert contests[c]['choice_function'] in ['IRV','plurality','supermajority'], \
                      'unsupported choice function ' + contests[c]['choice_function'] + ' in ' \
                      + c + ' contest'
            assert contests[c]['n_winners'] <= len(contests[c]['candidates']), \
                'fewer candidates than winners in ' + c + ' contest'
            assert len(contests[c]['reported_winners']) == contests[c]['n_winners'], \
                'number of reported winners does not equal n_winners in ' + c + ' contest'
            for w in contests[c]['reported_winners']:
                assert w in contests[c]['candidates'], \
                    'reported winner ' + w + ' is not a candidate in ' + c + 'contest'
            if contests[c]['choice_function'] in ['IRV','supermajority']:
                assert contests[c]['n_winners'] == 1, \
                    contests[c]['choice_function'] + ' can have only 1 winner in ' + c + ' contest'
            if contests[c]['choice_function'] == 'IRV':
                assert contests[c]['assertion_file'], 'IRV contest ' + c + ' requires an assertion file'
            if contests[c]['choice_function'] == 'supermajority':
                assert contests[c]['share_to_win'] >= 0.5, \
                    'super-majority contest requires winning at least 50% of votes in ' + c + ' contest'

    @classmethod
    def find_sample_size(
                         cls, contests: dict, sample_size_function: callable, use_style: bool=True, 
                         polling: bool=False, cvr_list: list=None, max_cards: int=None):
        '''
        Find initial sample size: maximum across assertions for all contests.

        Parameters:
        -----------
        contests: dict of dicts
        assertion: dict of dicts
        sample_size_function: callable
            takes three parameters: margin, risk limit, cards; returns a sample size
            cards is read from the contest dict.
        use_style: bool
            use style information to target the sample?
        polling: bool
            is this a polling audit? (False for comparison audit)
        cvr_list: list
            list of CVR objects
        max_cards: int
            upper bound on the true number of cards


        Returns:
        --------
        total_sample_size: int
            sample size expected to be adequate to confirm all assertions for all contests
        contest_sample_size: dict of ints
            sample sizes expected to be adequate to confirm each contest (keys are contests)
        '''
        sample_sizes = {c:0 for c in contests.keys()}
        if use_style and cvr_list is None:
            raise ValueError("use_style is True but cvr_list was not provided.")
        if use_style:
            for cvr in cvr_list:
                cvr.p=0
            for c in contests:
                risk = contests[c]['risk_limit']
                cards = contests[c]['cards']
                contest_sample_size = 0
                for a in contests[c]['assertions']:
                    margin = contests[c]['assertions'][a].margin
                    upper_bound = contests[c]['assertions'][a].upper_bound
                    u =  upper_bound if polling else 2/(2-margin/upper_bound)
                    contest_sample_size = np.max([contest_sample_size, sample_size_function(risk, margin, cards, u)])
                sample_sizes[c] = contest_sample_size
                # set the sampling probability for each card that contains the contest
                for cvr in cvr_list:
                    if cvr.has_contest(c):
                        cvr.p = np.maximum(contest_sample_size / contests[c]['cards'], cvr.p)

            total_sample_size = np.sum(np.array([x.p for x in cvr_list]))
        else:
            if max_cards is None:
                raise ValueError("use_style is False but max_cards was not provided.")
            cards = max_cards
            for c in contests:
                contest_sample_size = 0
                risk = contests[c]['risk_limit']
                for a in contests[c]['assertions']:
                    margin = contests[c]['assertions'][a].margin
                    upper_bound = contests[c]['assertions'][a].upper_bound
                    u = upper_bound if polling else 2/(1-margin/upper_bound)
                    contest_sample_size = np.max([contest_sample_size, sample_size_function(risk, margin, cards, u)])
                sample_sizes[c] = contest_sample_size
            total_sample_size = np.max(np.array(sample_sizes.values))
        return total_sample_size, sample_sizes

    @classmethod
    def prep_comparison_sample(cls, mvr_sample, cvr_sample, sample_order):
        '''
        prepare the MVRs and CVRs for comparison by putting the MVRs into the same (random) order
        in which the CVRs were selected

        conduct data integrity checks.

        Side-effects: sorts the mvr sample into the same order as the cvr sample

        Parameters
        ----------
        mvr_sample: list of CVR objects
            the manually determined votes for the audited cards
        cvr_sample: list of CVR objects
            the electronic vote record for the audited cards
        sample_order: dict
            dict to look up selection order of the cards. Keys are card identifiers. Values are dicts
            containing "selection_order" (which draw yielded the card) and "serial" (the card's original position)

        Returns
        -------
        '''
        mvr_sample.sort(key = lambda x: sample_order[x.id]["selection_order"])
        cvr_sample.sort(key = lambda x: sample_order[x.id]["selection_order"])
        assert len(cvr_sample) == len(mvr_sample),\
            "Number of cvrs ({}) and number of mvrs ({}) differ".format(len(cvr_sample), len(mvr_sample))
        for i in range(len(cvr_sample)):
            assert mvr_sample[i].id == cvr_sample[i].id, \
                f'Mismatch between id of cvr ({cvr_sample[i].id}) and mvr ({mvr_sample[i].id})'

    @classmethod
    def prep_polling_sample(cls, mvr_sample: list, sample_order: dict):
        '''
        Put the mvr sample back into the random selection order.

        Only about the side effects.

        Parameters
        ----------
        mvr_sample: list
            list of CVR objects
        sample_order: dict of dicts
            dict to look up selection order of the cards. Keys are card identifiers. Values are dicts
            containing "selection_order" (which draw yielded the card) and "serial" (the card's original position)

        Returns
        -------
        only side effects: mvr_sample is reordered
        '''
        mvr_sample.sort(key= lambda x: sample_order[x.id]["selection_order"])

    @classmethod
    def sort_cvr_sample_num(cls, cvr_list: list):
        '''
        Sort cvr_list by sample_num

        Only about the side effects.

        Parameters
        ----------
        cvr_list: list
            list of CVR objects

        Returns
        -------
        only side effects: cvr_list is ordered by sample_num
        '''
        cvr_list.sort(key = lambda x: x.sample_num)
        return True

    @classmethod
    def consistent_sampling(
                            cls, cvr_list: list, contests: dict, sample_size_dict: dict, 
                            sampled_cvr_indices: list=None) -> list:
        '''
        Sample CVR ids for contests to attain sample sizes in sample_size_dict

        Assumes that phantoms have already been generated and sample_num has been assigned
        to every CVR, including phantoms

        Parameters
        ----------
        cvr_list: list
            list of CVR objects
        contests: dict
            dict of contests
        sample_size_dict: dict
            dict of sample sizes by contest
        sampled_cvr_indices: list
            indices of cvrs already in the sample

        Returns
        -------
        sampled_cvr_indices: list
            locations of CVRs to sample
        '''
        current_sizes = defaultdict(int)
        contest_in_progress = lambda c: (current_sizes[c] < sample_size_dict[c])
        if sampled_cvr_indices is None:
            sampled_cvr_indices = []
        else:
            for sam in sampled_cvr_indices:
                for c in contests:
                    current_sizes[c] += (1 if cvr_list[sam].has_contest(c) else 0)
        sorted_cvr_indices = [i+1 for i, cv in sorted(enumerate(cvr_list), key = lambda x: x[1].sample_num)]
        inx = len(sampled_cvr_indices)
        while any([contest_in_progress(c) for c in contests]):
            if any([(contest_in_progress(c) and cvr_list[sorted_cvr_indices[inx]-1].has_contest(c)) for c in contests]):
                sampled_cvr_indices.append(sorted_cvr_indices[inx])
                for c in contests:
                    current_sizes[c] += (1 if cvr_list[sorted_cvr_indices[inx]-1].has_contest(c) else 0)
            inx += 1
        for i in range(len(cvr_list)):
            if i in sampled_cvr_indices:
                cvr_list[i].sampled = True
        return sampled_cvr_indices

    @classmethod
    def new_sample_size(
                        cls, contests: dict, mvr_sample: list, cvr_sample: list=None, cvr_list: list=None, 
                        use_style: bool=True, polling: bool=False, 
                        risk_function: callable=(lambda x, m, N, u:TestNonnegMean.alpha_mart(x)[1]),
                        quantile: float=0.5, reps: int=200, seed: int=1234567890) -> typing.Tuple[int, dict]:
        '''
        Estimate sample size for each contest and overall to allow the audit to complete,
        if discrepancies continue at the rate already observed.
        
        For comparison audits only.

        Uses simulations. For speed, uses the numpy.random Mersenne Twister instead of cryptorandom.

        Parameters:
        -----------
        contests: dict of dicts
            the contest data structure. outer keys are contest identifiers; inner keys are assertions

        mvr_sample: list of CVR objects
            the manually ascertained voter intent from sheets, including entries for phantoms

        cvr_sample: list of CVR objects
            the cvrs for the same sheets. For

        use_style: bool
            If True, use style information inferred from CVRs to target the sample on cards that contain
            each contest. Otherwise, sample from all cards.

        risk_function: callable
            function to calculate the p-value from overstatement_assorter values.
            Should take three arguments, the sample x, the margin m, and the number of cards N.

        quantile: float
            estimated quantile of the sample size to return

        reps: int
            number of replications to use to estimate the quantile

        seed: int
            seed for the Mersenne Twister prng

        Returns:
        --------
        new_size: int
            new sample size
        sams: array of ints
            array of all sizes found in the simulation
        '''
        if use_style and cvr_list is None:
            raise ValueError("use_style is True but cvr_list was not provided.")
        if use_style:
            for cvr in cvr_list:
                if cvr.in_sample():
                    cvr.p=1
                else:
                    cvr.p=0
        prng = np.random.RandomState(seed=seed)
        sample_sizes = {c:np.zeros(reps) for c in contests.keys()}
        #set dict of old sample sizes for each contest
        old_sizes = {c:0 for c in contests.keys()}
        for c in contests:
            old_sizes[c] = np.sum(np.array([cvr.in_sample() for cvr in cvr_list if cvr.has_contest(c)]))
        for r in range(reps):
            for c in contests:
                new_size = 0
                cards = contests[c]['cards']
                #raise an error or warning if the error rate implies the reported outcome is wrong
                for a in contests[c]['assertions']:
                    if not contests[c]['assertions'][a].proved:
                        p = contests[c]['assertions'][a].p_value
                        margin = contests[c]['assertions'][a].margin
                        upper_bound = contests[c]['assertions'][a].upper_bound
                        u = upper_bound if polling else 2/(2-margin/upper_bound)
                        if cvr_sample:
                            d = [contests[c]['assertions'][a].overstatement_assorter(mvr_sample[i], cvr_sample[i],\
                                contests[c]['assertions'][a].margin, use_style=use_style) for i in range(len(mvr_sample))]
                        else:
                            d = [contests[c]['assertions'][a].assort(mvr_sample[i], use_style=use_style) \
                                 for i in range(len(mvr_sample))]
                        while p > contests[c]['risk_limit'] and new_size < cards:
                            one_more = sample_by_index(len(d), 1, prng=prng)[0]
                            d.append(d[one_more-1])
                            p = risk_function(d, margin, cards, u)
                        new_size = np.max([new_size, len(d)])
                sample_sizes[c][r] = new_size
        new_sample_size_quantiles = {c:int(np.quantile(sample_sizes[c], quantile) - old_sizes[c]) for c in sample_sizes.keys()}
        if cvr_list:
            for cvr in cvr_list:
                for c in contests:
                    if cvr.has_contest(c) and not cvr.in_sample():
                        cvr.p = np.max(new_sample_size_quantiles[c] / (contests[c]['cards'] - old_sizes[c]), cvr.p)
            total_sample_size = np.round(np.sum(np.array([x.p for x in cvr_list])))
        else:
            total_sample_size = np.max(np.array(new_sample_size_quantiles.values))
        return total_sample_size, new_sample_size_quantiles

    @classmethod
    def summarize_status(cls, contests):
        '''
        Determine whether the audit of individual assertions, contests, and the whole election are finished.

        Prints a summary.

        Parameters:
        -----------
        contests: dict of dicts
            dict of contest information

        Returns:
        --------
        done: boolean
            is the audit finished?'''
        done = True
        for c in contests:
            print("p-values for assertions in contest {}".format(c))
            cpmax = 0
            for a in contests[c]['assertions']:
                cpmax = np.max([cpmax,contests[c]['assertions'][a].p_value])
                print(a, contests[c]['assertions'][a].p_value)
            if cpmax <= contests[c]['risk_limit']:
                print("\ncontest {} AUDIT COMPLETE at risk limit {}. Attained risk {}".format(\
                    c, contests[c]['risk_limit'], cpmax))
            else:
                done = False
                print("\ncontest {} audit INCOMPLETE at risk limit {}. Attained risk {}".format(\
                    c, contests[c]['risk_limit'], cpmax))
                print("assertions remaining to be proved:")
                for a in contests[c]['assertions']:
                    if contests[c]['assertions'][a].p_value > contests[c]['risk_limit']:
                        print("{}: current risk {}".format(a, contests[c]['assertions'][a].p_value))
        return done

    @classmethod
    def write_audit_parameters(
                               cls, log_file, seed, replacement, risk_function,
                               max_cards, n_cvrs, manifest_cards, phantom_cards, error_rate, contests):
        '''
        Write audit parameters to log_file as a json structure

        Parameters:
        ---------
        log_file: string
            filename to write to

        seed: string
            seed for the PRNG for sampling ballots

        risk_function: string
            name of the risk-measuring function used in the audit

        error_rate: float
            expected rate of 1-vote overstatements

        contests: dict of dicts
            contest-specific information for the audit

        Returns:
        --------
        no return value
        '''
        out = {"seed": seed,
               "replacement": replacement,
               "risk_function": risk_function,
               "max_cards": int(max_cards),
               "n_cvrs": int(n_cvrs),
               "manifest_cards": int(manifest_cards),
               "phantom_cards": int(phantom_cards),
               "error_rate": error_rate,
               "contests": contests
              }
        with open(log_file, 'w') as f:
            f.write(json.dumps(out, cls=NpEncoder))

    @classmethod
    def trim_ints(cls, x):
        '''
        turn int64 into an int

        Parameters
        ----------
        x: int64

        Returns
        -------
        int(x): int
       '''
        if isinstance(x, np.int64):
            return int(x)
        else:
            raise TypeError

class NpEncoder(json.JSONEncoder):
    def default(self, obj):
        if isinstance(obj, np.integer):
            return int(obj)
        if isinstance(obj, np.floating):
            return float(obj)
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        if isinstance(obj, Assertion):
            return obj.__str__()
        return super(NpEncoder, self).default(obj)

################
# Unit tests


def test_initial_sample_size():
    max_cards = int(10**3)
    risk_function_vec = lambda x, m, N, u: TestNonnegMean.kaplan_kolmogorov(x, N=N, t=1/2, g=0.1)[1]
    n_det = Utils.initial_sample_size(risk_function_vec, max_cards, margin=0.1, t=1/2, u=1, alpha=0.05,
                                               polling=False, error_rate=0.001)
    n_rand = Utils.initial_sample_size(risk_function_vec, max_cards, margin=0.1, t=1/2, u=1, alpha=0.05,
                                                polling=False, error_rate=0.001, reps=100)
    print(f'test_initial_sample_size: {n_det=}, {n_rand=}')


def test_initial_sample_size_alpha():
    max_cards = int(10**3)
    margin = 0.1
    upper_bound = 1
    u = 2/(2-margin/upper_bound)
    error_rate = 0.001
    m = (1 - error_rate*upper_bound/margin)/(2*upper_bound/margin - 1)
    estim = lambda x, N, mu, eta, u: TestNonnegMean.shrink_trunc(x=x, N=N, mu=mu, eta=eta, u=u, 
                                                        c=(eta-mu)/2, d=100, f=2, minsd=10**-6)
    risk_fn_vec = lambda x, N, mu, u: TestNonnegMean.alpha_mart(x, N=N, t=mu, eta=(m+1)/2, u=u, estim=estim)[1]
    n_det = Utils.initial_sample_size(risk_fn_vec, max_cards, margin=0.1, t=1/2, u=1, alpha=0.05,
                                               polling=False, error_rate=0.001)
    n_rand = Utils.initial_sample_size(risk_fn_vec, max_cards, margin=0.1, t=1/2, u=1, alpha=0.05,
                                                polling=False, error_rate=0.001, reps=100)
    print(f'test_initial_sample_size_alpha: {n_det=}, {n_rand=}')


def test_initial_sample_size_KW():
    # Test the initial_sample_size on the Kaplan-Wald risk function
    g = 0
    alpha = 0.05
    error_rate = 0.01
    N = int(10**4)
    margin = 0.1
    upper_bound = 1
    u = 2/(2-margin/upper_bound)
    m = (1 - error_rate*upper_bound/margin)/(2*upper_bound/margin - 1)
    one_over = 1/3.8 # 0.5/(2-margin)
    clean = 1/1.9    # 1/(2-margin)

    risk_fn_vec = lambda x, margin, N, u: TestNonnegMean.kaplan_wald(x, t=1/2, g=g, random_order=False)[1]
    # first test
    bias_up = False
    sam_size = Utils.initial_sample_size(risk_fn_vec, N, margin=m, t=1/2, u=u, alpha=alpha,
                                                  polling=False, error_rate=error_rate,
                                                  reps=None, bias_up=bias_up, quantile=0.5, seed=1234567890)
    sam_size_0 = 59 # math.ceil(math.log(20)/math.log(2/1.9)), since this is < 1/error_rate
    np.testing.assert_almost_equal(sam_size, sam_size_0)
    # second test
    bias_up = True
    sam_size = Utils.initial_sample_size(risk_fn_vec, N, margin=m, t=1/2, u=u, alpha=alpha,
                                                  polling=False, error_rate=error_rate, 
                                                  reps=None, bias_up=bias_up, quantile=0.5, seed=1234567890)
    sam_size_1 = 72 # (1/1.9)*(2/1.9)**71 = 20.08
    np.testing.assert_almost_equal(sam_size, sam_size_1)
    # third test
    sam_size = Utils.initial_sample_size(risk_fn_vec, N, margin=m, t=1/2, u=u, alpha=alpha,
                                                  polling=False, error_rate=error_rate, 
                                                  reps=1000, bias_up=bias_up, quantile=0.5, seed=1234567890)
    np.testing.assert_array_less(sam_size_0, sam_size+1) # crude test, but ballpark
    np.testing.assert_array_less(sam_size, sam_size_1+1) # crude test, but ballpark

def test_consistent_sampling():
    cvrs = [CVR(id="1", votes={"city_council": {"Alice": 1}, "measure_1": {"yes": 1}}, phantom=False), \
            CVR(id="2", votes={"city_council": {"Bob": 1},   "measure_1": {"yes": 1}}, phantom=False), \
            CVR(id="3", votes={"city_council": {"Bob": 1},   "measure_1": {"no": 1}}, phantom=False), \
            CVR(id="4", votes={"city_council": {"Charlie": 1}}, phantom=False), \
            CVR(id="5", votes={"city_council": {"Doug": 1}}, phantom=False), \
            CVR(id="6", votes={"measure_1": {"no": 1}}, phantom=False)
            ]
    prng = SHA256(1234567890)
    CVR.assign_sample_nums(cvrs, prng)
    contests = {'city_council': ' ', 'measure_1': ' '}
    sample_cvr_indices = Utils.consistent_sampling(cvrs, contests, {'city_council': 3, 'measure_1': 3})
    assert sample_cvr_indices == [5, 4, 6, 1, 2]


if __name__ == "__main__":
    test_initial_sample_size()
    test_initial_sample_size_KW()
    test_initial_sample_size_alpha()
    test_consistent_sampling()
