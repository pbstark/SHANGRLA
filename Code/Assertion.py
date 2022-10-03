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
import Utils

class Assertion:
    '''
    Objects and methods for SHANGRLA assertions about election outcomes

    An _assertion_ is a statement of the form
      "the average value of this assorter applied to the ballots is greater than 1/2"
    An _assorter_ maps votes to nonnegative numbers not exceeding some upper bound `u`
    '''

    # supported json assertion types for imported assertions
    JSON_ASSERTION_TYPES = (WINNER_ONLY:= "WINNER_ONLY", 
                            IRV_ELIMINATION:= "IRV_ELIMINATION")  
    
    def __init__(
                 self, contest: object=None, assorter: callable=None, margin: float=None, 
                 upper_bound: float=1, risk_function: callable=None, 
                 p_value: float=1, p_history: list=[], proved: bool=False, sample_size=None):
        '''
        assorter() should produce a float in [0, upper_bound]
        risk_function() should produce a float in [0, 1]

        Parameters
        ----------

        contest: object
            identifier of the contest to which the assorter is relevant
        assorter: callable
            the assorter for the assertion
        margin: float
            the assorter margin
        risk_function: callable
            the function to find the p-value of the hypothesis that the assertion is true, i.e., that the 
            assorter mean is <=1/2
        p_value: float
            the current p-value for the complementary null hypothesis that the assertion is false
        p_history: list
            the history of p-values, sample by sample. Generally, it is valid only for sequential risk-measuring
            functions.
        proved: boolean
            has the complementary null hypothesis been rejected?
        sample_size: int
            estimated total sample size to complete the audit of this assertion

        '''
        self.assorter = assorter
        self.contest = contest
        self.margin = margin
        self.upper_bound = upper_bound
        self.risk_function = risk_function
        self.p_value = p_value
        self.p_history = p_history
        self.proved = proved
        self.sample_size = sample_size

    def __str__(self):
        return (f'contest: {self.contest} margin: {self.margin} upper bound: {self.upper_bound} '
                f'risk function: {self.risk_function} p-value: {self.p_value} '
                f'p-history length: {len(self.p_history)} proved: {self.proved} sample_size: {self.sample_size}'
               )

    def assort(self, cvr):
        return self.assorter.assort(cvr)

    def min_p(self):
        return min(self.p_history)

    def assorter_mean(self, cvr_list, use_style=True):
        '''
        find the mean of the assorter applied to a list of CVRs

        Parameters
        ----------
        cvr_list: list
            a list of cast-vote records
        use_style: Boolean
            does the audit use card style information? If so, apply the assorter only to CVRs
            that contain the contest in question.

        Returns
        -------
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

        Parameters
        ----------
        cvr_list: list of CVRs
            a list of cast-vote records
        use_style: Boolean
            does the audit use card style information? If so, apply the assorter only to CVRs
            that contain the contest in question.

        Returns
        -------
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
    
    def overstatement_assorter_margin(
                                      self, assorter_margin: float=None, one_vote_overstatement_rate: float=0,
                                      cvr_list: list=None) -> float:
        '''
        find the overstatement assorter margin corresponding to an assumed rate of 1-vote overstatements
        
        Parameters
        ----------        
        assorter_margin: float
            the margin for the underlying "raw" assorter. If this is not provided, calculates it from the CVR list
        one_vote_overstatement_rate: float
            the assumed rate of one-vote overstatement errors in the CVRs
        cvr_list: list
            CVRs to calculate the assorter margin. Only used if assorter_margin is None
    
        Returns
        -------
        the overstatement assorter margin implied by the reported margin and the assumed rate of one-vote overstatements
        '''
        if assorter_margin is None:
            if cvr_list:
                assorter_margin = self.assorter_margin(cvr_list)
            else:
                raise ValueError("must provide either assorter_margin or cvr_list")
        u = self.upper_bound
        return (1-r*u/assorter_margin)/(2*u/assorter_margin-1)
    
    def overstatement_assorter_mean(
                                    self, assorter_margin: float=None, one_vote_overstatement_rate: float=0,
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
            
        Parameters
        ----------
        assorter_margin: float
            the margin of the raw assorter
        one_vote_overstatement_rate: float
            assumed rate of one-vote overstatements
        cvr_list: list
            list of CVR objects to calculate the assorter margin, if the assorter margin was not provided
        
        Returns
        -------
        overstatement assorter mean implied by the assorter mean and the assumed rate of 1-vote overstatements
        
        '''
        if assorter_margin is None:
            if cvr_list:
                assorter_margin = self.assorter_margin(cvr_list)
            else:
                raise ValueError("must provide either assorter_margin or cvr_list")
        return (1-r/2)/(2-assorter_margin/self.upper_bound)
    

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

        Parameters
        ----------
        mvr: Cvr
            the manual interpretation of voter intent
        cvr: Cvr
            the machine-reported cast vote record

        Returns
        -------
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

    def overstatement_assorter(self, mvr: list=None, cvr: list=None, use_style=True) -> float:
        '''
        assorter that corresponds to normalized overstatement error for an assertion

        If `use_style = True`, then if the CVR contains the contest but the MVR does not,
        that is considered to be an overstatement, because the ballot is presumed to contain
        the contest.

        If `use_style == False`, then if the CVR contains the contest but the MVR does not,
        the MVR is considered to be a non-vote in the contest.

        Parameters:
        -----------
        mvr: Cvr
            the manual interpretation of voter intent
        cvr: Cvr
            the machine-reported cast vote record. 

        Returns:
        --------
        over: float
            (1-o/u)/(2-v/u), where
                o is the overstatement
                u is the upper bound on the value the assorter assigns to any ballot
                v is the assorter margin
        '''
        return (1-self.overstatement(mvr, cvr, use_style)/self.assorter.upper_bound)/(2-self.margin/self.assorter.upper_bound)
    
    def find_margin(self, cvr_list: list=None, use_style=False):
        '''
        find and set the assorter margin
        
        Parameters
        ----------
        cvr_list: list
            cvrs from which the sample will be drawn
        use_style: bool
            is the sample drawn only from ballots that should contain the contest?
            
        Returns
        -------
        nothing
        
        Side effects
        ------------
        sets assorter.margin
        
        '''
        amean =self.assorter_mean(cvr_list, use_style=use_style)
        if amean < 1/2:
            warnings.warn(f"assertion {a} not satisfied by CVRs: mean value is {amean}")
        self.margin = 2*amean-1
                
                
    def make_overstatement(self, overs: float, cvr_list: list=None, use_style: bool=False) -> float:
        '''
        return the numerical value corresponding to an overstatement of `overs` times the assorter upper bound `u`
        
        Parameters
        ----------
        overs: float
            the multiple of `u`
        cvr_list: list of CVR objects
            the cvrs. Only used if the assorter margin has not been set
        use_style: bool
            flag to use style information. Only used if the assorter margin has not been set
        
        Returns
        -------
        the numerical value corresponding to an overstatement of that multiple
        
        Side effects
        ------------
        sets the assorter's margin if it had not been set
        '''
        if not self.margin:
            self.find_margin(cvrs, use_style=use_style)
        return (1-overs/self.assorter.upper_bound)/(2-self.margin/self.assorter.upper_bound)
                

    def initial_sample_size(
                            self, error_rate: float=0, reps: int=None, bias_up: bool=True, 
                            quantile: float=0.5, seed: int=1234567890) -> int:
        '''
        Estimate sample size needed to reject the null hypothesis that the assorter mean is <=1/2,
        for the specified risk function, given the margin and--for comparison audits--assumptions 
        about the rate of overstatement errors.

        This function is for a single assorter.

        Implements two strategies if audit_style==BALLOT_COMPARISON:

            1. If reps is not None, uses simulations to estimate the `quantile` quantile
            of sample size required. The simulations use the numpy Mersenne Twister PRNG.

            2. If reps is None, puts discrepancies into the sample in a deterministic way, starting
            with a discrepancy in the first item if bias_up == True, then including an additional discrepancy after
            every int(1/error_rate) items in the sample. 
            "Frontloading" the errors (bias_up == True) should make this slightly conservative on average


        Parameters:
        -----------
        error_rate: float
            assumed rate of 1-vote overstatements for ballot-level comparison audit.
            Ignored if audit_style==POLLING
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
        alpha = self.contest['risk_limit']
        assert alpha > 0 and alpha < 1/2, f'{alpha=} is not in (0, 1/2)'
        assert self.margin > 0, f'Margin {self.margin} is nonpositive'
sample_size(
                    cls, risk_fn: callable, x: np.array, N: int, t: float=1/2, alpha: float=0.05, 
                    reps: int=None, prefix: bool=False, quantile: float=0.5, **kwargs) -> int:
        if self.contest['audit_type'] == Contest.BALLOT_COMPARISON:        
            clean = self.make_overstatement(overs=0)          # assumes margin has been set
            one_vote_over = self.make_overstatement(overs=1)  # assumes margin has been set
            if reps is None:     # allocate errors deterministically
                offset = 0 if bias_up else 1
                x = clean*np.ones(N)
                if error_rate > 0:
                    for k in range(N):
                        x[k] = (one_vote_over if (k+offset) % int(1/error_rate) == 0 else x[k])
                p = self.risk_function.sample_size(x)[1]
                crossed = (p<=alpha)
                sam_size = int(N if np.sum(crossed)==0 else (np.argmax(crossed)+1))
            else:                # estimate the quantile by simulation
                prng = np.random.RandomState(seed)  # use the Mersenne Twister for speed
                sams = np.zeros(int(reps))
                for r in range(reps):
                    x = clean*np.ones(N)
                    inx = (prng.random(size=N) <= error_rate)  # randomly allocate errors
                    x[inx] = one_vote_over
                    p = self.risk_function.test(x)[1]
                    crossed = (p<=alpha)
                    sams[r] = N if np.sum(crossed)==0 else (np.argmax(crossed)+1)
                sam_size = int(np.quantile(sams, quantile))
        elif self.audit_type == Contest.POLLING:     # ballot-polling audit
            if reps is None:
                raise ValueError('estimating ballot-polling sample size requires setting `reps`')
            else: 
                x = np.zeros(N)
                nonzero = math.floor((margin+1)/2)
                x[0:nonzero] = np.ones(nonzero)
                prng = np.random.RandomState(seed)  # use the Mersenne Twister for speed
                sams = np.zeros(int(reps))
                for r in range(reps):
                    x = prng.permutation(x)
                    p = self.risk_function.test(x)[1]
                    crossed = (p <= alpha)
                    sams[r] = N if np.sum(crossed)==0 else (np.argmax(crossed)+1)
                sam_size = int(np.quantile(sams, quantile))
        else:
            raise NotImplementedError(f'audit type {audit_type} not implemented')
        return sam_size

    @classmethod
    def make_plurality_assertions(
                                  cls, contest, winners, losers, audit_type: str=Contest.POLLING, 
                                  risk_function: callable=None):
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
        audit_type: str
            audit_type in Contest.AUDIT_TYPES
        risk_function: callable
            risk function for the contest

        Returns:
        --------
        a dict of Assertions

        '''
        if audit_type not in Contest.AUDIT_TYPES:
            raise ValueError(f'audit type {audit_type} not implemented')
        assertions = {}
        for winr in winners:
            for losr in losers:
                wl_pair = winr + ' v ' + losr
                assertions[wl_pair] = Assertion(contest, Assorter(contest=contest, 
                                      assort = lambda c, contest=contest, winr=winr, losr=losr:
                                      (CVR.as_vote(CVR.get_vote_from_cvr(contest, winr, c))
                                      - CVR.as_vote(CVR.get_vote_from_cvr(contest, losr, c))
                                      + 1)/2, upper_bound = 1), audit_type=audit_type, 
                                      risk_function=risk_function)
        return assertions

    @classmethod
    def make_supermajority_assertion(
                                     cls, contest, winner, losers, share_to_win: float=1/2, 
                                     audit_type: str=Contest.POLLING, risk_function: callable=None):
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
        audit_type: str
            audit_type in Contest.AUDIT_TYPES
        risk_function: callable
            risk function for the contest

        Returns:
        --------
        a dict containing one Assertion

        '''
        if audit_type not in Contest.AUDIT_TYPES:
            raise ValueError(f'audit type {audit_type} not implemented')
        assert share_to_win < 1, f"share_to_win is {share_to_win} but must be less than 1"

        assertions = {}
        wl_pair = winner + ' v all'
        cands = losers.copy()
        cands.append(winner)
        assertions[wl_pair] = Assertion(contest, \
                                 Assorter(contest=contest, 
                                          assort = lambda c, contest=contest: 
                                                CVR.as_vote(CVR.get_vote_from_cvr(contest, winner, c))/(2*share_to_win) 
                                                if CVR.has_one_vote(contest, cands, c) else 1/2,
                                          upper_bound = 1/(2*share_to_win)), 
                                 risk_function=risk_function)
        return assertions

    @classmethod
    def make_assertions_from_json(
                                  cls, contest, candidates, json_assertions, audit_type: str=Contest.POLLING, 
                                  risk_function: callable=None):
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
        audit_type: str
            audit_type in Contest.AUDIT_TYPES
        risk_function: callable
            risk function for the contest

        Returns:
        --------
        dict of assertions for each assertion specified in 'json_assertions'.
        '''
        if audit_type not in Contest.AUDIT_TYPES:
            raise ValueError(f'audit type {audit_type} not implemented')
        assertions = {}
        for assrtn in json_assertions:
            winr = assrtn['winner']
            losr = assrtn['loser']

            if assrtn['assertion_type'] == WINNER_ONLY:
                # CVR is a vote for the winner only if it has the
                # winner as its first preference
                winner_func = lambda v, contest=contest, winr=winr: 1 \
                              if CVR.get_vote_from_cvr(contest, winr, v) == 1 else 0

                # CVR is a vote for the loser if they appear and the
                # winner does not, or they appear before the winner
                loser_func = lambda v, contest=contest, winr=winr, losr=losr: \
                             CVR.rcv_lfunc_wo(contest, winr, losr, v)

                wl_pair = winr + ' v ' + losr
                assertions[wl_pair] = Assertion(contest, 
                                                Assorter(contest=contest, winner=winner_func, 
                                                   loser=loser_func, upper_bound=1), 
                                                risk_function=risk_function)

            elif assrtn['assertion_type'] == IRV_ELIMINATION:
                # Context is that all candidates in 'eliminated' have been
                # eliminated and their votes distributed to later preferences
                elim = [e for e in assrtn['already_eliminated']]
                remn = [c for c in candidates if c not in elim]
                # Identifier for tracking which assertions have been proved
                wl_given = winr + ' v ' + losr + ' elim ' + ' '.join(elim)
                assertions[wl_given] = Assertion(contest, Assorter(contest=contest, 
                                       assort = lambda v, contest=contest, winr=winr, losr=losr, remn=remn:
                                       ( CVR.rcv_votefor_cand(contest, winr, remn, v)
                                       - CVR.rcv_votefor_cand(contest, losr, remn, v) +1)/2,
                                       upper_bound = 1), risk_function=risk_function)
            else:
                raise NotImplemented(f'JSON assertion type {assertn["assertion_type"]} not implemented.')
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
            risk_function = contests[c]['risk_function']  ### FIX ME! risk_function needs to be initialized
            if scf == Contest.PLURALITY:
                contests[c]['assertions'] = Assertion.make_plurality_assertions(c, winrs, losrs, 
                                                                                risk_function=risk_function)
            elif scf == Contest.SUPERMAJORITY:
                contests[c]['assertions'] = Assertion.make_supermajority_assertion(c, winrs[0], losrs,
                                                    contests[c]['share_to_win'], risk_function=risk_function)
            elif scf == Contest.IRV:
                # Assumption: contests[c]['assertion_json'] yields list assertions in JSON format.
                contests[c]['assertions'] = Assertion.make_assertions_from_json(c, contests[c]['candidates'],
                                                    contests[c]['assertion_json'], risk_function=risk_function)
            else:
                raise NotImplementedError(f'Social choice function {scf} is not implemented.')
        return True

    @classmethod
    def set_all_margins(cls, contests: dict, cvr_list: list, use_style: bool):
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
                contests[c]['assertions'][a].margin = (margin:= contests[c]['assertions'][a].find_margin(use_style=use_style))
                contests[c]['margins'].update({a: margin})
                min_margin = np.min([min_margin, margin])
        return min_margin

    @classmethod
    def set_all_p_values(cls, contests: dict, mvr_sample: list, cvr_sample: list=None) -> float :
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
            audit_type = contests[c]['audit_type']
            use_style = contests[c]['use_style']
            for a in contests[c]['assertions']:
                asrt = contests[c]['assertions'][a]
                margin = asrt.margin
                upper_bound = asrt.upper_bound
                if audit_type == 'BALLOT_COMPARISON':
                    d = [asrt.overstatement_assorter(mvr_sample[i], cvr_sample[i],
                                margin, use_style=use_style) for i in range(len(mvr_sample)) 
                                if ((not use_style) or cvr_sample[i].has_contest(c))]
                    u = 2/(2-margin/upper_bound)
                elif audit_type=='POLLING':         # polling audit. Assume style information is irrelevant
                    d = [asrt.assort(mvr_sample[i]) for i in range(len(mvr_sample))]
                    u = upper_bound
                else:
                    raise NotImplementedError(f'audit type {audit_type} not implemented')
                contests[c]['assertions'][a].p_value, contests[c]['assertions'][a].p_history = \
                                                asrt.risk_function.test(d)
                contests[c]['assertions'][a].proved = ((
                                                contests[c]['assertions'][a].p_value <= contests[c]['risk_limit']) 
                                                or contests[c]['assertions'][a].proved)
                contests[c]['p_values'].update({a: contests[c]['assertions'][a].p_value})
                contests[c]['proved'].update({a: int(contests[c]['assertions'][a].proved)})
                contest_max_p = np.max([contest_max_p, contests[c]['assertions'][a].p_value])
            contests[c].update({'max_p': contest_max_p})
            p_max = np.max([p_max, contests[c]['max_p']])
        return p_max

    @classmethod
    def find_all_sample_sizes(
                              cls, contests: dict, use_style: bool=True, 
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
            raise ValueError("use_style==True but cvr_list was not provided.")
        if use_style:
            # initialize sampling probabilities
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
                    contest_sample_size = np.max([contest_sample_size, a.initial_sample_size(])
                sample_sizes[c] = contest_sample_size
                # set the sampling probability for each card that contains the contest
                for cvr in cvr_list:
                    if cvr.has_contest(c):
                        cvr.p = np.maximum(contest_sample_size / contests[c]['cards'], cvr.p)

            total_sample_size = np.sum(np.array([x.p for x in cvr_list]))
        else:
            if max_cards is None:
                raise ValueError("use_style==False but max_cards was not provided.")
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
    def find_all_new_sample_sizes(
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
            raise ValueError("use_style==True but cvr_list was not provided.")
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

    def __init__(self, contest: dict=None, assort: callable=None, winner: str=None, loser: str=None, upper_bound: float=1):
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



class Contest:
    '''
    Objects and methods for contests. 
    '''
    
    # social choice functions
    SOCIAL_CHOICE_FUNCTIONS = (PLURALITY:= 'PLURALITY', 
                               SUPERMAJORITY:= 'SUPERMAJORITY', 
                               IRV:= 'IRV')
    
    AUDIT_TYPES = (POLLING:= 'POLLING', BALLOT_COMPARISON:= 'BALLOT_COMPARISON') 
    # TO DO: BATCH_COMPARISON, STRATIFIED, HYBRID, ...

    ATTRIBUTES = ('id','name','risk_limit','cards','choice_function','n_winners','candidates','reported_winners',
                  'assertion_file','audit_type','risk_function','use_style')
    
    def __init__(
                 self, id: object=None, name: str=None, risk_limit: float=0.05, cards: int=0, 
                 choice_function: str=PLURALITY,
                 n_winners: int=1, candidates: list=None, reported_winners: list=None,
                 assertion_file: str=None, audit_type: str=Assertion.BALLOT_COMPARISON,
                 risk_function: callable=None,  use_style: bool=True):
        self.id = id
        self.name = name
        self.risk_limit = risk_limit
        self.cards = cards
        self.choice_function = choice_function
        self.n_winners = n_winners
        self.candidates = candidates
        self.reported_winners = reported_winners
        self.assertion_file = assertion_file
        self.audit_type = audit_type
        self.risk_function = risk_function
        self.use_style = use_style

    def initial_sample_size(self, **kwargs) -> int:
        '''
        Find the initial sample size to confirm the contest at its risk limit.
        
        To perform the calculation, 
        
        Parameters
        ----------
        '''
        sam_size = 0
        for c in self.assertions:
            c.sample_size = c.
            
        
                            
    def __str__(self):
                            

    @classmethod
    def from_dict(cls, d: dict) -> dict:
        '''
        define a dict of contest objects from a dict of dicts, each containing data for one contest
        '''
        contest_dict = {}
        for c in d:
            contest_dict[c] = {}
            for att in ATTRIBUTES:
                contest_dict[c][att] = d[c].get(att)
                
    
    
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
                    + 1)/2, upper_bound = 1), )
    aVb.margin=0.2
    assert aVb.overstatement_assorter(mvrs[0], cvrs[0], use_style=True) == 1/1.8
    assert aVb.overstatement_assorter(mvrs[0], cvrs[0], use_style=False) == 1/1.8

    assert aVb.overstatement_assorter(mvrs[1], cvrs[0], use_style=True) == 0
    assert aVb.overstatement_assorter(mvrs[1], cvrs[0], use_style=False) == 0

    aVb.margin=0.3
    assert aVb.overstatement_assorter(mvrs[0], cvrs[1], use_style=True) == 2/1.7
    assert aVb.overstatement_assorter(mvrs[0], cvrs[1], use_style=False) == 2/1.7

    aVb.margin=0.1
    assert aVb.overstatement_assorter(mvrs[2], cvrs[0], use_style=True) == 0.5/1.9
    assert aVb.overstatement_assorter(mvrs[2], cvrs[0], use_style=False) == 0.5/1.9


def test_assorter_mean():
    pass # [FIX ME]


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

########
if __name__ == "__main__":

    test_supermajority_assorter()
    test_rcv_assorter()
    test_assorter_mean()
    test_overstatement_assorter()
    test_overstatement()
    test_make_plurality_assertions()
    
    test_initial_sample_size()
    test_initial_sample_size_KW()
    test_initial_sample_size_alpha()