import math
import numpy as np
import scipy as sp
import json
import csv
import warnings
import typing
from numpy import testing
from collections import OrderedDict, defaultdict
from CVR import CVR

class Assertion:
    '''
    Objects and methods for SHANGRLA assertions about election outcomes

    An _assertion_ is a statement of the form
      "the average value of this assorter applied to the ballots is greater than 1/2"
    An _assorter_ maps votes to nonnegative numbers not exceeding some upper bound `u`
    '''

    JSON_ASSERTION_TYPES = ["WINNER_ONLY", "IRV_ELIMINATION"]  # supported json assertion types for imported assertions

    def __init__(self, contest: object=None, assorter: callable=None, margin: float=None, 
                 upper_bound: float=1, p_value: float=1, p_history: list=[], proved: bool=False):
        '''
        The assorter is callable; should produce a non-negative real.

        Parameters
        ----------

        contest: object
            identifier of the contest to which the assorter is relevant
        assorter: callable
            the assorter for the assertion
        margin: float
            the assorter margin
        p_value: float
            the current p-value for the complementary null hypothesis that the assertion is false
        p_history: list
            the history of p-values, sample by sample. Generally, it is valid only for sequential risk-measuring
            functions.
        proved: boolean
            has the complementary null hypothesis been rejected?

        '''
        self.assorter = assorter
        self.contest = contest
        self.margin = margin
        self.upper_bound = upper_bound
        self.p_value = p_value
        self.p_history = p_history
        self.proved = proved

    def __str__(self):
        return (f'contest: {self.contest} margin: {self.margin} upper bound: {self.upper_bound} '
                f'p-value: {self.p_value} p-history length: {len(self.p_history)} proved: {self.proved}'
               )

    def assort(self, cvr):
        return self.assorter.assort(cvr)

    def min_p(self):
        return min(self.p_history)

    def assorter_mean(self, cvr_list, use_style=True):
        '''
        find the mean of the assorter applied to a list of CVRs

        Parameters:
        -----------
        cvr_list: list
            a list of cast-vote records
        use_style: Boolean
            does the audit use card style information? If so, apply the assorter only to CVRs
            that contain the contest in question.

        Returns:
        --------N
        mean: float
            the mean value of the assorter over the list of cvrs. If use_style, ignores CVRs that
            do not contain the contest.
        '''
        if use_style:
            filtr = lambda c: c.has_contest(self.contest)
        else:
            filtr = lambda c: True
        return np.mean([self.assorter.assort(c) for c in cvr_list if filtr(c)])

    def assorter_sum(self, cvr_list, use_style=True):
        '''
        find the sum of the assorter applied to a list of CVRs

        Parameters:
        ----------
        cvr_list: list of CVRs
            a list of cast-vote records
        use_style: Boolean
            does the audit use card style information? If so, apply the assorter only to CVRs
            that contain the contest in question.

        Returns:
        ----------
        sum: float
            sum of the value of the assorter over a list of CVRs. If use_style, ignores CVRs that
            do not contain the contest.
        '''
        if use_style:
            filtr = lambda c: c.has_contest(self.contest)
        else:
            filtr = lambda c: True
        return np.sum([self.assorter.assort(c) for c in cvr_list if filtr(c)])

    def assorter_margin(self, cvr_list, use_style=True):
        '''
        find the margin for a list of Cvrs.
        By definition, the margin is twice the mean of the assorter, minus 1.

        Parameters:
        ----------
        cvr_list: list
            a list of cast-vote records

        Returns:
        ----------
        margin: float
        '''
        return 2*self.assorter_mean(cvr_list, use_style=use_style)-1
    
    def overstatement_assorter_margin(self, assorter_margin: float=None, one_vote_overstatement_rate: float=0,
                                     cvr_list: list=None) -> float:
        '''
        find the overstatement assorter margin corresponding to an assumed rate of 1-vote overstatements
        
        Parameters:
        
        assorter_margin: float
            the margin for the underlying "raw" assorter. If this is not provided, calculates it from the CVR list
        one_vote_overstatement_rate: float
            the assumed rate of one-vote overstatement errors in the CVRs
        cvr_list: list
            CVRs to calculate the assorter margin. Only used if assorter_margin is None
        '''
        if assorter_margin is None:
            if cvr_list:
                assorter_margin = self.assorter_margin(cvr_list)
            else:
                raise ValueError("must provide either assorter_margin or cvr_list")
        u = self.upper_bound
        return (1-r*u/assorter_margin)/(2*u/assorter_margin-1)
    
    def overstatement_assorter_mean(self, assorter_margin: float=None, one_vote_overstatement_rate: float=0,
                                     cvr_list: list=None) -> float:
        '''
        find the overstatement assorter mean corresponding to an assumed rate of 1-vote overstatements
        
        Parameters:
        
        assorter_margin: float
            the margin for the underlying "raw" assorter. If not provided, calculated from the CVR list
        one_vote_overstatement_rate: float
            the assumed rate of one-vote overstatement errors in the CVRs
        cvr_list: list
            CVRs to calculate the assorter margin. Only used if assorter_margin is None
        '''
        if assorter_margin is None:
            if cvr_list:
                assorter_margin = self.assorter_margin(cvr_list)
            else:
                raise ValueError("must provide either assorter_margin or cvr_list")
        u = self.upper_bound
        return (1-r/2)/(2-assorter_margin/u)
    

    def overstatement(self, mvr, cvr, use_style=True):
        '''
        overstatement error for a CVR compared to the human reading of the ballot

        If use_style, then if the CVR contains the contest but the MVR does
        not, treat the MVR as having a vote for the loser (assort()=0)

        If not use_style, then if the CVR contains the contest but the MVR does not,
        the MVR is considered to be a non-vote in the contest (assort()=1/2).

        Phantom CVRs and MVRs are treated specially:
            A phantom CVR is considered a non-vote in every contest (assort()=1/2).
            A phantom MVR is considered a vote for the loser (i.e., assort()=0) in every
            contest.

        Parameters:
        -----------
        mvr: Cvr
            the manual interpretation of voter intent
        cvr: Cvr
            the machine-reported cast vote record

        Returns:
        --------
        overstatement: float
            the overstatement error
        '''
        if not use_style:
            overstatement = self.assorter.assort(cvr)\
                            - (self.assorter.assort(mvr) if not mvr.phantom else 0)
        elif use_style:
            if cvr.has_contest(self.contest):    # make_phantoms() assigns contests but not votes to phantom CVRs
                if cvr.phantom:
                    cvr_assort = 1/2
                else:
                    cvr_assort = self.assorter.assort(cvr)
                if mvr.phantom or not mvr.has_contest(self.contest):
                    mvr_assort = 0
                else:
                    mvr_assort = self.assorter.assort(mvr)
                overstatement = cvr_assort - mvr_assort
            else:
                raise ValueError("Assertion.overstatement: use_style==True but CVR does not contain the contest")
        return overstatement

    def overstatement_assorter(self, mvr: list=None, cvr: list=None, margin: float=None, use_style=True):
        '''
        assorter that corresponds to normalized overstatement error for an assertion

        If use_style, then if the CVR contains the contest but the MVR does not,
        that is considered to be an overstatement, because the ballot is presumed to contain
        the contest.

        If not use_style, then if the CVR contains the contest but the MVR does not,
        the MVR is considered to be a non-vote in the contest.

        Parameters:
        -----------
        mvr: Cvr
            the manual interpretation of voter intent
        cvr: Cvr
            the machine-reported cast vote record
        margin: float
            2*(assorter applied to all CVRs) - 1, the assorter margin

        Returns:
        --------
        over: float
            (1-o/u)/(2-v/u), where
                o is the overstatement
                u is the upper bound on the value the assorter assigns to any ballot
                v is the assorter margin
        '''
        return (1-self.overstatement(mvr, cvr, use_style)/self.assorter.upper_bound)\
                /(2-margin/self.assorter.upper_bound)
    
    def initial_sample_size(
                            self, risk_fn: callable, polling: bool=False, error_rate: float=0, 
                            reps: int=None, bias_up: bool=True, quantile: float=0.5, seed: int=1234567890) -> int:
        '''
        Estimate sample size needed to reject the null hypothesis that the assorter mean is <=1/2,
        for the specified risk function, given the margin
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
                p = risk_function(x, margin, N, u)
                crossed = (p<=alpha)
                sam_size = int(N if np.sum(crossed)==0 else (np.argmax(crossed)+1))
            else:                # estimate the quantile by simulation
                prng = np.random.RandomState(seed)  # use the Mersenne Twister for speed
                sams = np.zeros(int(reps))
                for r in range(reps):
                    pop = clean*np.ones(N)
                    inx = (prng.random(size=N) <= error_rate)  # randomly allocate errors
                    pop[inx] = one_vote_over
                    p = risk_function(pop, margin, N, u)
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
    def make_plurality_assertions(cls, contest, winners, losers):
        '''
        Construct assertions that imply the winner(s) got more votes than the loser(s).

        The assertions are that every winner beat every loser: there are
        len(winners)*len(losers) pairwise assertions in all.

        Parameters:
        -----------
        winners: list
            list of identifiers of winning candidate(s)
        losers: list
            list of identifiers of losing candidate(s)

        Returns:
        --------
        a dict of Assertions

        '''
        assertions = {}
        for winr in winners:
            for losr in losers:
                wl_pair = winr + ' v ' + losr
                assertions[wl_pair] = Assertion(contest, Assorter(contest=contest, \
                                      assort = lambda c, contest=contest, winr=winr, losr=losr:\
                                      ( CVR.as_vote(CVR.get_vote_from_cvr(contest, winr, c)) \
                                      - CVR.as_vote(CVR.get_vote_from_cvr(contest, losr, c)) \
                                      + 1)/2, upper_bound = 1))
        return assertions

    @classmethod
    def make_supermajority_assertion(cls, contest, winner, losers, share_to_win):
        '''
        Construct assertion that winner got >= share_to_win \in (0,1) of the valid votes

        An equivalent condition is:

        (votes for winner)/(2*share_to_win) + (invalid votes)/2 > 1/2.

        Thus the correctness of a super-majority outcome--where share_to_win >= 1/2--can
        be checked with a single assertion.

        share_to_win < 1/2 might be useful for some social choice functions, including
        primaries where candidates who receive less than some threshold share are
        eliminated.

        A CVR with a mark for more than one candidate in the contest is considered an
        invalid vote.

        Parameters:
        -----------
        contest: string
            identifier of contest to which the assertion applies
        winner :
            identifier of winning candidate
        losers: list
            list of identifiers of losing candidate(s)
        share_to_win: float
            fraction of the valid votes the winner must get to win

        Returns:
        --------
        a dict containing one Assertion

        '''
        assert share_to_win < 1, f"share_to_win is {share_to_win} but must be less than 1"

        assertions = {}
        wl_pair = winner + ' v all'
        cands = losers.copy()
        cands.append(winner)
        assertions[wl_pair] = Assertion(contest, \
                                 Assorter(contest=contest, assort = lambda c, contest=contest: \
                                 CVR.as_vote(CVR.get_vote_from_cvr(contest, winner, c))/(2*share_to_win) \
                                 if CVR.has_one_vote(contest, cands, c) else 1/2,\
                                 upper_bound = 1/(2*share_to_win) ))
        return assertions

    @classmethod
    def make_assertions_from_json(cls, contest, candidates, json_assertions):
        '''
        dict of Assertion objects from a RAIRE-style json representations of assertions.

        The assertion_type for each assertion must be one of the JSON_ASSERTION_TYPES
        (class constants).

        Parameters:
        -----------
        contest: string
            identifier of contest to which the assorter applies

        candidates :
            list of identifiers for all candidates in relevant contest.

        json_assertions:
            Assertions to be tested for the relevant contest.

        Returns:
        --------
        dict of assertions for each assertion specified in 'json_assertions'.
        '''
        assertions = {}
        for assrtn in json_assertions:
            winr = assrtn['winner']
            losr = assrtn['loser']

            # Is this a 'winner only' assertion
            if assrtn['assertion_type'] not in Assertion.JSON_ASSERTION_TYPES:
                raise ValueError("assertion type " + assrtn['assertion_type'])

            elif assrtn['assertion_type'] == "WINNER_ONLY":
                # CVR is a vote for the winner only if it has the
                # winner as its first preference
                winner_func = lambda v, contest=contest, winr=winr: 1 \
                              if CVR.get_vote_from_cvr(contest, winr, v) == 1 else 0

                # CVR is a vote for the loser if they appear and the
                # winner does not, or they appear before the winner
                loser_func = lambda v, contest=contest, winr=winr, losr=losr: \
                             CVR.rcv_lfunc_wo(contest, winr, losr, v)

                wl_pair = winr + ' v ' + losr
                assertions[wl_pair] = Assertion(contest, Assorter(contest=contest, winner=winner_func, \
                                                loser=loser_func, upper_bound=1))

            elif assrtn['assertion_type'] == "IRV_ELIMINATION":
                # Context is that all candidates in 'eliminated' have been
                # eliminated and their votes distributed to later preferences
                elim = [e for e in assrtn['already_eliminated']]
                remn = [c for c in candidates if c not in elim]
                # Identifier for tracking which assertions have been proved
                wl_given = winr + ' v ' + losr + ' elim ' + ' '.join(elim)
                assertions[wl_given] = Assertion(contest, Assorter(contest=contest, \
                                       assort = lambda v, contest=contest, winr=winr, losr=losr, remn=remn: \
                                       ( CVR.rcv_votefor_cand(contest, winr, remn, v) \
                                       - CVR.rcv_votefor_cand(contest, losr, remn, v) +1)/2,\
                                       upper_bound = 1))
            else:
                raise NotImplemented('JSON assertion type %s not implemented. ' \
                                      % assertn['assertion_type'])
        return assertions

    @classmethod
    def make_all_assertions(cls, contests: dict):
        '''
        Construct all the assertions to audit the contests and add the assertions to the contest dict

        Parameters
        ----------
        contests: dict
            the contest-level data

        Returns
        -------
        True

        Side Effects
        ------------
        adds the list of assertions relevant to each contest to the contest dict under the key 'assertions'

        '''
        for c in contests:
            scf = contests[c]['choice_function']
            winrs = contests[c]['reported_winners']
            losrs = [cand for cand in contests[c]['candidates'] if cand not in winrs]
            if scf == 'plurality':
                contests[c]['assertions'] = Assertion.make_plurality_assertions(c, winrs, losrs)
            elif scf == 'supermajority':
                contests[c]['assertions'] = Assertion.make_supermajority_assertion(c, winrs[0], losrs, \
                                  contests[c]['share_to_win'])
            elif scf == 'IRV':
                # Assumption: contests[c]['assertion_json'] yields list assertions in JSON format.
                contests[c]['assertions'] = Assertion.make_assertions_from_json(c, contests[c]['candidates'], \
                    contests[c]['assertion_json'])
            else:
                raise NotImplementedError("Social choice function " + scf + " is not supported")
        return True

    @classmethod
    def find_margins(cls, contests: dict, cvr_list: list, use_style: bool):
        '''
        Find all the assorter margins in a set of Assertions. Updates the dict of dicts of assertions
        and the contest dict.

        Appropriate only if cvrs are available. Otherwise, base margins on the reported results.

        This function is primarily about side-effects on the assertions in the contest dict.

        Parameters:
        -----------
        contests: dict of contest data, including assertions
        cvr_list: list
            list of cvr objects
        use_style: bool
            flag indicating the sample will use style information to target the contest

        Returns:
        --------
        min_margin: float
            smallest margin in the audit
        '''
        min_margin = np.infty
        for c in contests:
            contests[c]['margins'] = {}
            for a in contests[c]['assertions']:
                amean = contests[c]['assertions'][a].assorter_mean(cvr_list, use_style=use_style)
                if amean < 1/2:
                    warnings.warn(f"assertion {a} not satisfied by CVRs: mean value is {amean}")
                margin = 2*amean-1
                contests[c]['assertions'][a].margin = margin
                contests[c]['margins'].update({a: margin})
                min_margin = np.min([min_margin, margin])
        return min_margin

    @classmethod
    def find_p_values(
                      cls, contests: dict, mvr_sample: list, cvr_sample: list=None, use_style: bool=False,
                      risk_function: callable=(lambda x, N, t, **kwargs: TestNonnegMean.kaplan_wald(x))) -> float :
        '''
        Find the p-value for every assertion and update assertions & contests accordingly

        update p_value, p_history, proved flag, the maximum p-value for each contest.

        Primarily about side-effects.

        Parameters:
        -----------
        contests: dict of dicts
            the contest data structure. outer keys are contest identifiers; inner keys are assertions

        mvr_sample: list of CVR objects
            the manually ascertained voter intent from sheets, including entries for phantoms

        cvr_sample: list of CVR objects
            the cvrs for the same sheets, for ballot-level comparison audits
            not needed for polling audits

        use_style: Boolean
            See documentation

        risk_function: function (class methods are not of type Callable)
            function to calculate the p-value from assorter values

        Returns:
        --------
        p_max: float
            largest p-value for any assertion in any contest

        Side-effects
        ------------
        Sets contest max_p to be the largest P-value of any assertion for that contest
        Updates p_value, p_history, and proved for every assertion

        '''
        if cvr_sample is not None:
            assert len(mvr_sample) == len(cvr_sample), "unequal numbers of cvrs and mvrs"
        p_max = 0
        for c in contests.keys():
            contests[c]['p_values'] = {}
            contests[c]['proved'] = {}
            contest_max_p = 0
            for a in contests[c]['assertions']:
                margin = contests[c]['assertions'][a].margin
                upper_bound = contests[c]['assertions'][a].upper_bound
                if cvr_sample: # comparison audit
                    d = [contests[c]['assertions'][a].overstatement_assorter(mvr_sample[i], cvr_sample[i],\
                         margin, use_style=use_style) for i in range(len(mvr_sample)) \
                         if ((not use_style) or cvr_sample[i].has_contest(c)) ]
                    u = 2/(2-margin/upper_bound)
                else:         # polling audit. Assume style information is irrelevant
                    d = [contests[c]['assertions'][a].assort(mvr_sample[i]) for i in range(len(mvr_sample))]
                    u = upper_bound
                contests[c]['assertions'][a].p_value, contests[c]['assertions'][a].p_history = \
                         risk_function(d, margin, contests[c]['cards'], u)
                # should the risk_function be called with the margin? FIX ME!
                contests[c]['assertions'][a].proved = \
                         (contests[c]['assertions'][a].p_value <= contests[c]['risk_limit']) \
                         or contests[c]['assertions'][a].proved
                contests[c]['p_values'].update({a: contests[c]['assertions'][a].p_value})
                contests[c]['proved'].update({a: int(contests[c]['assertions'][a].proved)})
                contest_max_p = np.max([contest_max_p, contests[c]['assertions'][a].p_value])
            contests[c].update({'max_p': contest_max_p})
            p_max = np.max([p_max, contests[c]['max_p']])
        return p_max

class Assorter:
    '''
    Class for generic Assorter.

    An assorter must either have an `assort` method or both `winner` and `loser` must be defined
    (in which case assort(c) = (winner(c) - loser(c) + 1)/2. )

    Class parameters:
    -----------------
    contest: string
        identifier of the contest to which this Assorter applies

    winner: callable
        maps a dict of selections into the value 1 if the dict represents a vote for the winner

    loser : callable
        maps a dict of selections into the value 1 if the dict represents a vote for the winner

    assort: callable
        maps dict of selections into float

    upper_bound: float
        a priori upper bound on the value the assorter assigns to any dict of selections

    The basic method is assort, but the constructor can be called with (winner, loser)
    instead. In that case,

        assort = (winner - loser + 1)/2

    '''

    def __init__(self, contest=None, assort=None, winner=None, loser=None, upper_bound = 1):
        '''
        Constructs an Assorter.

        If assort is defined and callable, it becomes the class instance of assort

        If assort is None but both winner and loser are defined and callable,
           assort is defined to be 1/2 if winner=loser; winner, otherwise


        Parameters:
        -----------
        assort: callable
            maps a dict of votes into [0, \infty)
        winner: callable
            maps a pattern into {0, 1}
        loser : callable
            maps a pattern into {0, 1}
        '''
        self.contest = contest
        self.winner = winner
        self.loser = loser
        self.upper_bound = upper_bound
        if assort is not None:
            assert callable(assort), "assort must be callable"
            self.assort = assort
        else:
            assert callable(winner), "winner must be callable if assort is None"
            assert callable(loser),  "loser must be callable if assort is None"
            self.assort = lambda cvr: (self.winner(cvr) - self.loser(cvr) + 1)/2

####### Unit tests
def test_make_plurality_assertions():
    winners = ["Alice","Bob"]
    losers = ["Candy","Dan"]
    asrtns = Assertion.make_plurality_assertions('AvB', winners, losers)
    assert asrtns['Alice v Candy'].assorter.assort(CVR.from_vote({"Alice": 1})) == 1
    assert asrtns['Alice v Candy'].assorter.assort(CVR.from_vote({"Bob": 1})) == 1/2
    assert asrtns['Alice v Candy'].assorter.assort(CVR.from_vote({"Candy": 1})) == 0
    assert asrtns['Alice v Candy'].assorter.assort(CVR.from_vote({"Dan": 1})) == 1/2

    assert asrtns['Alice v Dan'].assorter.assort(CVR.from_vote({"Alice": 1})) == 1
    assert asrtns['Alice v Dan'].assorter.assort(CVR.from_vote({"Bob": 1})) == 1/2
    assert asrtns['Alice v Dan'].assorter.assort(CVR.from_vote({"Candy": 1})) == 1/2
    assert asrtns['Alice v Dan'].assorter.assort(CVR.from_vote({"Dan": 1})) == 0

    assert asrtns['Bob v Candy'].assorter.assort(CVR.from_vote({"Alice": 1})) == 1/2
    assert asrtns['Bob v Candy'].assorter.assort(CVR.from_vote({"Bob": 1})) == 1
    assert asrtns['Bob v Candy'].assorter.assort(CVR.from_vote({"Candy": 1})) == 0
    assert asrtns['Bob v Candy'].assorter.assort(CVR.from_vote({"Dan": 1})) == 1/2

    assert asrtns['Bob v Dan'].assorter.assort(CVR.from_vote({"Alice": 1})) == 1/2
    assert asrtns['Bob v Dan'].assorter.assort(CVR.from_vote({"Bob": 1})) == 1
    assert asrtns['Bob v Dan'].assorter.assort(CVR.from_vote({"Candy": 1})) == 1/2
    assert asrtns['Bob v Dan'].assorter.assort(CVR.from_vote({"Dan": 1})) == 0

def test_supermajority_assorter():
    losers = ["Bob","Candy"]
    share_to_win = 2/3
    assn = Assertion.make_supermajority_assertion("AvB","Alice", losers, share_to_win)

    votes = CVR.from_vote({"Alice": 1})
    assert assn['Alice v all'].assorter.assort(votes) == 3/4, "wrong value for vote for winner"

    votes = CVR.from_vote({"Bob": True})
    assert assn['Alice v all'].assorter.assort(votes) == 0, "wrong value for vote for loser"

    votes = CVR.from_vote({"Dan": True})
    assert assn['Alice v all'].assorter.assort(votes) == 1/2, "wrong value for invalid vote--Dan"

    votes = CVR.from_vote({"Alice": True, "Bob": True})
    assert assn['Alice v all'].assorter.assort(votes) == 1/2, "wrong value for invalid vote--Alice & Bob"

    votes = CVR.from_vote({"Alice": False, "Bob": True, "Candy": True})
    assert assn['Alice v all'].assorter.assort(votes) == 1/2, "wrong value for invalid vote--Bob & Candy"


def test_rcv_assorter():
    import json
    with open('Data/334_361_vbm.json') as fid:
        data = json.load(fid)

        assertions = {}
        for audit in data['audits']:
            cands = [audit['winner']]
            for elim in audit['eliminated']:
                cands.append(elim)

            all_assertions = Assertion.make_assertions_from_json('AvB', cands, audit['assertions'])

            assertions[audit['contest']] = all_assertions

        assorter = assertions['334']['5 v 47'].assorter
        votes = CVR.from_vote({'5': 1, '47': 2})
        assert(assorter.assort(votes) == 1)

        votes = CVR.from_vote({'47': 1, '5': 2})
        assert(assorter.assort(votes) == 0)

        votes = CVR.from_vote({'3': 1, '6': 2})
        assert(assorter.assort(votes) == 0.5)

        votes = CVR.from_vote({'3': 1, '47': 2})
        assert(assorter.assort(votes) == 0)

        votes = CVR.from_vote({'3': 1, '5': 2})
        assert(assorter.assort(votes) == 0.5)

        assorter = assertions['334']['5 v 3 elim 1 6 47'].assorter
        votes = CVR.from_vote({'5': 1, '47': 2})
        assert(assorter.assort(votes) == 1)

        votes = CVR.from_vote({'47': 1, '5': 2})
        assert(assorter.assort(votes) == 1)

        votes = CVR.from_vote({'6': 1, '1': 2, '3': 3, '5': 4})
        assert(assorter.assort(votes) == 0)

        votes = CVR.from_vote({'3': 1, '47': 2})
        assert(assorter.assort(votes) == 0)

        votes = CVR.from_vote({})
        assert(assorter.assort(votes) == 0.5)

        votes = CVR.from_vote({'6': 1, '47': 2})
        assert(assorter.assort(votes) == 0.5)

        votes = CVR.from_vote({'6': 1, '47': 2, '5': 3})
        assert(assorter.assort(votes) == 1)

        assorter = assertions['361']['28 v 50'].assorter
        votes = CVR.from_vote({'28': 1, '50': 2})
        assert(assorter.assort(votes) == 1)
        votes = CVR.from_vote({'28': 1})
        assert(assorter.assort(votes) == 1)
        votes = CVR.from_vote({'50': 1})
        assert(assorter.assort(votes) == 0)

        votes = CVR.from_vote({'27': 1, '28': 2})
        assert(assorter.assort(votes) == 0.5)

        votes = CVR.from_vote({'50': 1, '28': 2})
        assert(assorter.assort(votes) == 0)

        votes = CVR.from_vote({'27': 1, '26': 2})
        assert(assorter.assort(votes) == 0.5)

        votes = CVR.from_vote({})
        assert(assorter.assort(votes) == 0.5)

        assorter = assertions['361']['27 v 26 elim 28 50'].assorter
        votes = CVR.from_vote({'27': 1})
        assert(assorter.assort(votes) == 1)

        votes = CVR.from_vote({'50': 1, '27': 2})
        assert(assorter.assort(votes) == 1)

        votes = CVR.from_vote({'28': 1, '50': 2, '27': 3})
        assert(assorter.assort(votes) == 1)

        votes = CVR.from_vote({'28': 1, '27': 2, '50': 3})
        assert(assorter.assort(votes) == 1)

        votes = CVR.from_vote({'26': 1})
        assert(assorter.assort(votes) == 0)

        votes = CVR.from_vote({'50': 1, '26': 2})
        assert(assorter.assort(votes) == 0)

        votes = CVR.from_vote({'28': 1, '50': 2, '26': 3})
        assert(assorter.assort(votes) == 0)

        votes = CVR.from_vote({'28': 1, '26': 2, '50': 3})
        assert(assorter.assort(votes) == 0)

        votes = CVR.from_vote({'50': 1})
        assert(assorter.assort(votes) == 0.5)
        votes = CVR.from_vote({})
        assert(assorter.assort(votes) == 0.5)

        votes = CVR.from_vote({'50': 1, '28': 2})
        assert(assorter.assort(votes) == 0.5)

        votes = CVR.from_vote({'28': 1, '50': 2})
        assert(assorter.assort(votes) == 0.5)


def test_overstatement():
    mvr_dict = [{'id': 1, 'votes': {'AvB': {'Alice':True}}},\
                {'id': 2, 'votes': {'AvB': {'Bob':True}}},
                {'id': 3, 'votes': {'AvB': {}}},\
                {'id': 4, 'votes': {'CvD': {'Elvis':True, 'Candy':False}}},\
                {'id': 'phantom_1', 'votes': {'AvB': {}}, 'phantom': True}]
    mvrs = CVR.from_dict(mvr_dict)

    cvr_dict = [{'id': 1, 'votes': {'AvB': {'Alice':True}}},\
                {'id': 2, 'votes': {'AvB': {'Bob':True}}},\
                {'id': 3, 'votes': {'AvB': {}}},\
                {'id': 4, 'votes': {'CvD': {'Elvis':True}}},\
                {'id': 'phantom_1', 'votes': {'AvB': {}}, 'phantom': True}]
    cvrs = CVR.from_dict(cvr_dict)

    winners = ["Alice"]
    losers = ["Bob"]

    aVb = Assertion("AvB", Assorter(contest="AvB", \
                    assort = lambda c, contest="AvB", winr="Alice", losr="Bob":\
                    ( CVR.as_vote(CVR.get_vote_from_cvr("AvB", winr, c)) \
                    - CVR.as_vote(CVR.get_vote_from_cvr("AvB", losr, c)) \
                    + 1)/2, upper_bound = 1))
    assert aVb.overstatement(mvrs[0], cvrs[0], use_style=True) == 0
    assert aVb.overstatement(mvrs[0], cvrs[0], use_style=False) == 0

    assert aVb.overstatement(mvrs[0], cvrs[1], use_style=True) == -1
    assert aVb.overstatement(mvrs[0], cvrs[1], use_style=False) == -1

    assert aVb.overstatement(mvrs[2], cvrs[0], use_style=True) == 1/2
    assert aVb.overstatement(mvrs[2], cvrs[0], use_style=False) == 1/2

    assert aVb.overstatement(mvrs[2], cvrs[1], use_style=True) == -1/2
    assert aVb.overstatement(mvrs[2], cvrs[1], use_style=False) == -1/2


    assert aVb.overstatement(mvrs[1], cvrs[0], use_style=True) == 1
    assert aVb.overstatement(mvrs[1], cvrs[0], use_style=False) == 1

    assert aVb.overstatement(mvrs[2], cvrs[0], use_style=True) == 1/2
    assert aVb.overstatement(mvrs[2], cvrs[0], use_style=False) == 1/2

    assert aVb.overstatement(mvrs[3], cvrs[0], use_style=True) == 1
    assert aVb.overstatement(mvrs[3], cvrs[0], use_style=False) == 1/2

    try:
        tst = aVb.overstatement(mvrs[3], cvrs[3], use_style=True)
        raise AssertionError('aVb is not contained in the mvr or cvr')
    except ValueError:
        pass
    assert aVb.overstatement(mvrs[3], cvrs[3], use_style=False) == 0

    assert aVb.overstatement(mvrs[4], cvrs[4], use_style=True) == 1/2
    assert aVb.overstatement(mvrs[4], cvrs[4], use_style=False) == 1/2
    assert aVb.overstatement(mvrs[4], cvrs[4], use_style=False) == 1/2
    assert aVb.overstatement(mvrs[4], cvrs[0], use_style=True) == 1
    assert aVb.overstatement(mvrs[4], cvrs[0], use_style=False) == 1
    assert aVb.overstatement(mvrs[4], cvrs[1], use_style=True) == 0
    assert aVb.overstatement(mvrs[4], cvrs[1], use_style=False) == 0


def test_overstatement_assorter():
    '''
    (1-o/u)/(2-v/u)
    '''
    mvr_dict = [{'id': 1, 'votes': {'AvB': {'Alice':True}}},\
                {'id': 2, 'votes': {'AvB': {'Bob':True}}},\
                {'id': 3, 'votes': {'AvB': {'Candy':True}}}]
    mvrs = CVR.from_dict(mvr_dict)

    cvr_dict = [{'id': 1, 'votes': {'AvB': {'Alice':True}}},\
                {'id': 2, 'votes': {'AvB': {'Bob':True}}}]
    cvrs = CVR.from_dict(cvr_dict)

    winners = ["Alice"]
    losers = ["Bob"]

    aVb = Assertion("AvB", Assorter(contest="AvB", \
                    assort = lambda c, contest="AvB", winr="Alice", losr="Bob":\
                    ( CVR.as_vote(CVR.get_vote_from_cvr("AvB", winr, c)) \
                    - CVR.as_vote(CVR.get_vote_from_cvr("AvB", losr, c)) \
                    + 1)/2, upper_bound = 1))
    assert aVb.overstatement_assorter(mvrs[0], cvrs[0], margin=0.2, use_style=True) == 1/1.8
    assert aVb.overstatement_assorter(mvrs[0], cvrs[0], margin=0.2, use_style=False) == 1/1.8

    assert aVb.overstatement_assorter(mvrs[1], cvrs[0], margin=0.2, use_style=True) == 0
    assert aVb.overstatement_assorter(mvrs[1], cvrs[0], margin=0.2, use_style=False) == 0

    assert aVb.overstatement_assorter(mvrs[0], cvrs[1], margin=0.3, use_style=True) == 2/1.7
    assert aVb.overstatement_assorter(mvrs[0], cvrs[1], margin=0.3, use_style=False) == 2/1.7

    assert aVb.overstatement_assorter(mvrs[2], cvrs[0], margin=0.1, use_style=True) == 0.5/1.9
    assert aVb.overstatement_assorter(mvrs[2], cvrs[0], margin=0.1, use_style=False) == 0.5/1.9


def test_assorter_mean():
    pass # [FIX ME]

########
if __name__ == "__main__":

    test_make_plurality_assertions()
    test_supermajority_assorter()
    test_rcv_assorter()
    test_assorter_mean()
    test_overstatement_assorter()
    test_overstatement()