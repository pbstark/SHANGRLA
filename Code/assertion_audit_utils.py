from __future__ import division, print_function
import math
import numpy as np
import scipy as sp
import pandas as pd
import json
import csv
import warnings
from numpy import testing

class Assertion:
    """
    Objects and methods for assertions about elections.
    An _assertion_ is a statement of the form 
      "the average value of this assorter applied to the ballots is greater than 1/2"
    The _assorter_ maps votes to nonnegative numbers not exceeding `upper_bound`
    """
    
    JSON_ASSERTION_TYPES = ["WINNER_ONLY", "IRV_ELIMINATION"]
    
    def __init__(self, contest = None, assorter = None):
        """
        The assorter is callable; should produce a non-negative real.
        """
        self.assorter = assorter
        self.contest = contest
        
    def set_assorter(self, assorter):
        self.assorter = assorter
        
    def get_assorter(self):
        return self.assorter

    def set_contest(self, contest):
        self.contest = contest
        
    def get_contest(self):
        return self.contest

    def assort(self, cvr):
        return self.assorter(cvr)
    
    def assorter_mean(self, cvr_list):
        """
        find the mean of the assorter values for a list of CVRs        
        Parameters:
        ----------
        cvr_list : list
            a list of cast-vote records
        
        Returns:
        ----------
        double
        """
        return np.mean(map(self.assorter, cvr_list))        
        
    def assorter_sum(self, cvr_list):
        """
        find the sum of the assorter values for a list of CVRs        
        Parameters:
        ----------
        cvr_list : list
            a list of cast-vote records
        
        Returns:
        ----------
        double
        """
        return np.sum(map(self.assorter, [k.votes for k in cvr_list]))   # BROKEN FIX ME! 

    def assorter_margin(self, cvr_list):
        """
        find the margin for a list of CVRs
        
        Parameters:
        ----------
        cvr_list : list
            a list of cast-vote records
        
        Returns:
        ----------
        margin : double
        """
        return 2*(self.assorter_mean(cvr_list)- 1)
        
    def margin_fraction(self, cvr_list):
        """
        find the diluted margin (as a fraction) for a list of CVRs
        
        Parameters:
        ----------
        cvr_list : list
            a list of cast-vote records
        
        Returns:
        ----------
        margin : double
        """
        return self.margin_votes(cvr_list)/len(cvr_list)
    
    def overstatement(self, mvr, cvr):
        """
        Find the overstatement error (in votes) in a CVR compared to the human 
        reading of the ballot
        
        Parameters:
        -----------
        mvr : Cvr
            the manual interpretation of voter intent
        cvr : Cvr
            the machine-reported cast vote record
            
        Returns:
        --------
        overstatement error
        """
        return self.assorter(cvr)-self.assorter(mvr)
    
    def overstatement_assorter(self, mvr, cvr, cvrs):
        """
        The assorter that corresponds to normalized overstatement error
        the overstatement error (in votes) in a CVR compared to the human 
        reading of the ballot
        
        Parameters:
        -----------
        mvr : Cvr
            the manual interpretation of voter intent
        cvr : Cvr
            the machine-reported cast vote record
        cvrs : list of Cvrs
            the entire list of CVRs for the contest
            
        Returns:
        --------
        overstatement error
        """
        return 1 - (self.assorter(cvr)-self.assorter(mvr))/(2*self.assorter_mean(cvrs)-1)
        
    
    def overstatements(self, mvr_list, cvr_list):
        """
        Count the discrepancies between human reading of a collection of ballots and the
        CVRs for those ballots
        
        Parameters:
        -----------
        mvr_list : list
            list of manually determined CVRs
        cvr_list : list
            list of the machine-reported CVRs
            
        Returns:
        --------
        tuple : number of ballots with overstatements of -2, -1, 1, and 2.
        
        """
        assert len(mvr_list) == len(cvr_list), "number of mvrs differs from number of cvrs"
        return np.array(map(self.overstatement, mvr_list, cvr_list))
        
      
    @classmethod
    def make_plurality_assertions(cls, contest, winners, losers):
        """
        Construct a set of assertions that imply that the winner(s) got more votes than the loser(s).
        
        The assertions are that every winner beat every loser: there are
        len(winners)*len(losers) pairwise assertions in all.
        
        Parameters:
        -----------
        winners : list
            list of identifiers of winning candidate(s)
        losers : list
            list of identifiers of losing candidate(s)
        
        Returns:
        --------
        a dict of Assertions
        
        """
        assertions = {}
        for winr in winners:
            for losr in losers:
                wl_pair = winr + ' v ' + losr                
                assertions[wl_pair] = Assertion(Assorter(contest=contest, \
                                      assort = lambda c, contest=contest, winr=winr, losr=losr:\
                                      ( CVR.as_vote(CVR.get_vote_from_votes(contest, winr, c)) \
                                      - CVR.as_vote(CVR.get_vote_from_votes(contest, losr, c)) \
                                      + 1)/2, upper_bound = 1))
        return assertions
    
    @classmethod
    def make_supermajority_assertion(cls, contest, winner, losers, share_to_win):
        """
        Construct a set of assertions that imply that the winner got at least a fraction 
        fraction_to_win of the valid votes.
        
        An equivalent condition is:
        
        (votes for winner)/(2*share_to_win) + (invalid votes)/2 > 1/2.
        
        Thus the correctness of a super-majority outcome can be checked with a single assertion.
        
        A CVR with a mark for more than one candidate in the contest is considered an invalid vote.
            
        Parameters:
        -----------
        contest : string
            identifier of contest to which the assertion applies
        winner : 
            identifier of winning candidate
        losers : list
            list of identifiers of losing candidate(s)
        share_to_win : double
            fraction of the valid votes the winner must get to win        
        
        Returns:
        --------
        a dict containing one Assertion
        
        """
        assert share_to_win > 1/2, "share_to_win must be at least 1/2"
        assert share_to_win < 1, "share_to_win must be less than 1"

        assertions = {}
        wl_pair = winner + ' v all'
        cands = losers.copy()
        cands.append(winner)
        assertions[wl_pair] = Assertion(Assorter(contest=contest, assort = lambda c, contest=contest: \
                                 CVR.as_vote(CVR.get_vote_from_votes(contest, winner, c))/(2*share_to_win) \
                                 if CVR.has_one_vote(contest, cands, c) else 1/2,\
                                 upper_bound = 1/(2*share_to_win) ))
        return assertions

    @classmethod
    def make_assertions_from_json(cls, contest, candidates, json_assertions):
        """
        Construct a dict of Assertion objects from a RAIRE-style json representation 
        of a list of assertions for a given contest.
        
        The assertion_type for each assertion must be one of the JSON_ASSERTION_TYPES (class constants)
        Each assertion should contain a winner and a 

        Parameters:
        -----------
        contest : string
            identifier of contest to which the assorter applies
            
        candidates : 
            list of identifiers for all candidates in relevant contest.

        json_assertions:
            Assertions to be tested for the relevant contest.

        Returns:
        --------
        A dict of assertions for each assertion specified in 'json_assertions'.
        """        
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
                winner_func = lambda v, contest=contest, winr=winr : 1 if CVR.get_vote_from_votes(contest, winr, v) == 1 else 0

                # CVR is a vote for the loser if they appear and the 
                # winner does not, or they appear before the winner
                loser_func = lambda v, contest=contest, winr=winr, losr=losr : CVR.rcv_lfunc_wo(contest, winr, losr, v)

                wl_pair = winr + ' v ' + losr
                assertions[wl_pair] = Assertion(Assorter(contest=contest, winner=winner_func, \
                                                loser=loser_func, upper_bound=1))

            elif assrtn['assertion_type'] == "IRV_ELIMINATION": 
                # Context is that all candidates in 'eliminated' have been
                # eliminated and their votes distributed to later preferences
                elim = [e for e in assrtn['already_eliminated']]
                remn = [c for c in candidates if c not in elim]
                # Identifier for tracking which assertions have been proved
                wl_given = winr + ' v ' + losr + ' elim ' + ' '.join(elim)
                assertions[wl_given] = Assertion(Assorter(contest=contest, \
                                       assort = lambda v, contest=contest, winr=winr, losr=losr, remn=remn : \
                                       ( CVR.rcv_votefor_cand(contest, winr, remn, v) \
                                       - CVR.rcv_votefor_cand(contest, losr, remn, v) +1)/2,\
                                       upper_bound = 1))
            else:
                raise NotImplemented('JSON assertion type %s not implemented. ' \
                                      % assertn['assertion_type'])
        return assertions
    
    @classmethod
    def make_all_assertions(cls, contests):
        """
        Construct all the assertions to audit the contests.
        
        Parameters:
        -----------
        contests : dict
            the contest-level data
        
        Returns:
        --------
        A dict of dicts of Assertion objects
        """
        all_assertions = {}
        for c in contests:
            scf = contests[c]['choice_function']
            winrs = contests[c]['reported_winners']
            losrs = [cand for cand in contests[c]['candidates'] if cand not in winrs]
            if scf == 'plurality':
                all_assertions[c] = Assertion.make_plurality_assertions(c, winrs, losrs)
            elif scf == 'supermajority':
                all_assertions[c] = Assertion.make_supermajority_assertion(c, winrs[0], losrs, \
                                  contests[c]['share_to_win'])
            elif scf == 'IRV':
                # Assumption: contests[c]['assertion_json'] yields list assertions in JSON format.
                all_assertions[c] = Assertion.make_assertions_from_json(c, contests[c]['candidates'], \
                    contests[c]['assertion_json'])
            else:
                raise NotImplementedError("Social choice function " + scf + " is not supported")
        return all_assertions

class Assorter:
    """
    Class for generic Assorter.
    
    An assorter must either have an `assort` method or both `winner` and `loser` must be defined
    (in which case assort(c) = (winner(c) - loser(c) + 1)/2. )
    
    Class parameters:
    -----------------
    contest : string
        identifier of the contest to which this Assorter applies
        
    winner : callable
        maps a dict of selections into the value 1 if the dict represents a vote for the winner   
        
    loser  : callable
        maps a dict of selections into the value 1 if the dict represents a vote for the winner
    
    assort : callable
        maps dict of selections into double
    
    upper_bound : double
        a priori upper bound on the value the assorter assigns to any dict of selections

    The basic method is assort, but the constructor can be called with (winner, loser)
    instead. In that case,
    
        assort = (winner - loser + 1)/2

    """
        
    def __init__(self, contest=None, assort=None, winner=None, loser=None, upper_bound = 1):
        """
        Constructs an Assorter.
        
        If assort is defined and callable, it becomes the class instance of assort
        
        If assort is None but both winner and loser are defined and callable,
           assort is defined to be 1/2 if winner=loser; winner, otherwise
           
        
        Parameters:
        -----------
        assort : callable
            maps a dict of votes into [0, \infty)
        winner : callable
            maps a pattern into {0, 1}
        loser  : callable
            maps a pattern into {0, 1}
        """   
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
            self.assort = lambda cvr: (self.winner(cvr["votes"][self.contest]) \
                                       - self.loser(cvr["votes"][self.contest]) + 1)/2 
    
    def set_winner(self, winner):
        self.winner = winner

    def get_winner(self):
        return(self.winner)

    def set_loser(self, loser):
        self.loser = loser

    def get_loser(self):
        return(self.loser)
    
    def set_assort(self, assort):
        self.assort = assort

    def get_assort(self):
        return(self.assort)

    def set_upper_bound(self, upper_bound):
        self.upper_bound = upper_bound
        
    def get_upper_bound(self):
        return self.upper_bound


class CVR:
    """
    Generic class for cast-vote records.
    
    The CVR class DOES NOT IMPOSE VOTING RULES. For instance, the social choice
    function might consider a CVR that contains two votes in a contest to be an overvote.
    
    Rather, a CVR is supposed to reflect what the ballot shows, even if the ballot does not
    contain a valid vote in one or more contests.
    
    Class method get_votefor returns the vote for a given candidate if the candidate is a 
    key in the CVR, or False if the candidate is not in the CVR. 
    
    This allows very flexible representation of votes, including ranked voting.
        
    For instance, in a plurality contest with four candidates, a vote for Alice (and only Alice)
    in a mayoral contest could be represented by any of the following:
            {"ID": "A-001-01", "votes": {"mayor": {"Alice": True}}}
            {"ID": "A-001-01", "votes": {"mayor": {"Alice": "marked"}}}
            {"ID": "A-001-01", "votes": {"mayor": {"Alice": 5}}}
            {"ID": "A-001-01", "votes": {"mayor": {"Alice": 1, "Bob": 0, "Candy": 0, "Dan": ""}}}
            {"ID": "A-001-01", "votes": {"mayor": {"Alice": True, "Bob": False}}}
    A CVR that contains a vote for Alice for "mayor" and a vote for Bob for "DA" could be represented as
            {"ID": "A-001-01", "votes": {"mayor": {"Alice": True}, "DA": {"Bob": True}}}
            
    Many methods in this class are defined for the "votes" portion of a contest within a CVR.
    For instance, bool(vote_for("Alice","mayor"))==True iff the CVR contains a vote for Alice
    in the contest named "mayor", and int(bool(vote_for("Alice","mayor")))==1 if the CVR 
    contains a vote for Alice in the contest named "mayor", and 0 otherwise.
                
    Ranked votes also have simple representation, e.g., if the CVR is
            {"ID": "A-001-01", "votes": {"mayor": {"Alice": 1, "Bob": 2, "Candy": 3, "Dan": ''}}}
    Then int(vote_for("Candy","mayor"))=3, Candy's rank in the "mayor" contest.
    
     
    Methods:
    --------
    
    get_votefor :  
         get_votefor(candidate, contest, votes) returns the value in the votes dict for the key `candidate`, or
         False if the candidate did not get a vote or the contest is not in the CVR
    set_votes :  
         updates the votes dict; overrides previous votes and/or creates votes for additional contests or candidates
    get_votes : returns complete votes dict for a contest
    get_ID : returns ballot ID
    set_ID : updates the ballot ID
        
    """
    
    def __init__(self, ID = {}, votes = {}):
        self.votes = votes
        self.ID = ID
        
    def get_votes(self):
        return self.votes
    
    def set_votes(self, votes):
        self.votes.update(votes)
            
    def get_ID(self):
        return self.ID
    
    def set_ID(self, ID):
        self.ID = ID
        
    def get_votefor(self, contest, candidate):
        return CVR.get_vote_from_votes(candidate, self.votes[contest])
    
    @classmethod
    def cvrs_to_json(cls, cvr):
        return json.dump(cvr)
    
    @classmethod
    def as_vote(cls, v):
        return int(bool(v))
    
    @classmethod
    def as_rank(cls, v):
        return int(v)
    
    @classmethod
    def get_vote_from_votes(cls, contest, candidate, votes):
        """
        Returns the vote for a candidate if the dict of votes contains a vote for that candidate
        in that contest; otherwise returns False
        
        Parameters:
        -----------
        contest : string
            identifier for the contest
        candidate : string
            identifier for candidate
        
        votes : dict
            a dict of votes 
        
        Returns:
        --------
        vote
        """
        return False if (contest not in votes or candidate not in votes[contest])\
               else votes[contest][candidate]
    
    @classmethod
    def get_vote_from_cvr(cls, contest, candidate, cvr):
        """
        Returns the vote for a candidate if the cvr contains a vote for that candidate; 
        otherwise returns False
        
        Parameters:
        -----------
        contest : string
            identifier for contest
        candidate : string
            identifier for candidate
        
        cvr : a CVR object
        
        Returns:
        --------
        vote
        """
        return False if (contest not in cvr.votes or candidate not in cvr.votes[contest])\
               else cvr.votes[contest][candidate]
    
    @classmethod
    def has_one_vote(cls, contest, candidates, votes):
        """
        Is there exactly one vote among the candidates in the contest?
        
        Parameters:
        -----------
        contest : string
            identifier of contest
        candidates : list
            list of identifiers of candidates
        
        Returns:
        ----------
        True if there is exactly one vote among those candidates in that contest, where a 
        vote means that the value for that key casts as boolean True.
        """
        votes = np.sum([0 if c not in votes[contest] else bool(votes[contest][c]) for c in candidates])
        return True if votes==1 else False
    
    @classmethod
    def rcv_lfunc_wo(cls, contest, winner, loser, vote):
        """
        Check whether vote is a vote for the loser with respect to a 'winner only' 
        assertion between the given 'winner' and 'loser'.  

        Parameters:
        -----------
        contest : string
            identifier for the contest
        winner : string
            identifier for winning candidate
        loser : string
            identifier for losing candidate

        vote : dict of dicts

        Returns:
        --------
        1 if the given vote is a vote for 'loser' and 0 otherwise
        """
        rank_winner = CVR.get_vote_from_votes(contest, winner, vote)
        rank_loser = CVR.get_vote_from_votes(contest, loser, vote)

        if not bool(rank_winner) and bool(rank_loser):
            return 1
        elif bool(rank_winner) and bool(rank_loser) and rank_loser < rank_winner:
            return 1
        else:
            return 0 

    @classmethod
    def rcv_votefor_cand(cls, contest, cand, remaining, vote):
        """
        Check whether 'vote' is a vote for the given candidate in the context
        where only candidates in 'remaining' remain standing.

        Parameters:
        -----------
        contest : string
            identifier of the contest
        cand : string
            identifier for candidate
        remaining : list
            list of identifiers of candidates still standing

        vote : dict of dicts

        Returns:
        --------
        1 if the given vote for the contest counts as a vote for 'cand' and 0 otherwise. Essentially,
        if you reduce the ballot down to only those candidates in 'remaining',
        and 'cand' is the first preference, return 1; otherwise return 0.
        """
        if not cand in remaining:
            return 0

        rank_cand = CVR.get_vote_from_votes(contest, cand, vote)

        if not bool(rank_cand):
            return 0
        else:
            for altc in remaining:
                if altc == cand:
                    continue
                rank_altc = CVR.get_vote_from_votes(contest, altc, vote)
                if bool(rank_altc) and rank_altc <= rank_cand:
                    return 0
            return 1 

class TestNonnegMean:
    r"""Tests of the hypothesis that the mean of a non-negative population is less than
        a threshold t.
        Several tests are implemented, all ultimately based on martingales:
            Kaplan-Markov
            Kaplan-Wald
            Kaplan-Kolmogorov
            Wald SPRT with replacement (only for binary-valued populations)
            Wald SPRT without replacement (only for binary-valued populations)
            Kaplan's martingale (KMart)        
    """
    
    TESTS = ['kaplan_markov','kaplan_wald','kaplan_kolmogorov','wald_sprt','kaplan_martingale']
    
    @classmethod
    def wald_sprt(cls, x, N, t = 1/2, p1=1, random_order = True):
        """
        Finds the p value for the hypothesis that the population 
        mean is less than or equal to t against the alternative that it is p1,
        for a binary population of size N.
       
        If N is finite, assumes the sample is drawn without replacement
        If N is infinite, assumes the sample is with replacement
       
        Parameters:
        -----------
        x : binary list, one element per draw. A list element is 1 if the 
            the corresponding trial was a success
        N : int
            population size for sampling without replacement, or np.infinity for 
            sampling with replacement
        t : double in (0,1)
            hypothesized fraction of ones in the population
        p1 : double in (0,1) greater than t
            alternative hypothesized fraction of ones in the population
        random_order : Boolean
            if the data are in random order, setting this to True can improve the power.
            If the data are not in random order, set to False
        """
        if any(xx not in [0,1] for xx in x):
            raise ValueError("Data must be binary")
        terms = np.ones(len(x))
        if np.isfinite(N):
            A = np.cumsum(np.insert(x, 0, 0)) # so that cumsum does the right thing
            for k in range(len(x)):
                if x[k] == 1.0:
                    if (N*t - A[k]) > 0:
                        terms[k] = np.max([N*p1 - A[k], 0])/(N*t - A[k])
                    else:
                        terms[k] = np.inf
                else:
                    if (N*(1-t) - k + 1 + A[k]) > 0:
                        terms[k] = np.max([(N*(1-p1) - k + 1 + A[k]), 0])/(N*(1-t) - k + 1 + A[k])
                    else:
                        terms[k] = np.inf
        else:
            terms[x==0] = (1-p1)/(1-t)
            terms[x==1] = p1/t
        return np.max(np.cumprod(terms)) if random_order else np.prod(terms)

    @classmethod
    def kaplan_markov(cls, x, t=1/2, g=0, random_order=True):
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
        x : array-like
            the sample
        t : double
            the null value of the mean
        g : double
            "padding" in case there any values in the sample are zero
        random_order : Boolean
            if the sample is in random order, it is legitimate to stop early, which 
            can yield a more powerful test. See above.
        
        Returns:
        --------
        p-value
        
        """       
        if any(x < 0):
            raise ValueError('Negative value in sample from a nonnegative population.')
        return np.min([1, np.min(np.cumprod((t+g)/(x+g))) if random_order else np.prod((t+g)/(x+g))])

    @classmethod
    def kaplan_wald(cls, x, t=1/2, g=0, random_order=True):
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
        x : array-like
            the sample
        t : double
            the null value of the mean
        g : double
            "padding" in case there any values in the sample are zero
        random_order : Boolean
            if the sample is in random order, it is legitimate to stop early, which 
            can yield a more powerful test. See above.

        Returns:
        --------
        p-value
       
        """       
        if any(x < 0):
            raise ValueError('Negative value in sample from a nonnegative population.')
        return np.min([1, 1/np.max(np.cumprod((1-g)*x/t + g)) if random_order \
                       else 1/np.prod((1-g)*x/t + g)])
    
    @classmethod
    def kaplan_kolmogorov(cls, x, N, t=1/2, random_order = True):
        '''
        p-value for the hypothesis that the mean of a nonnegative population with N
        elements is t. The alternative is that the mean is less than t.
        If the random sample x is in the order in which the sample was drawn, it is
        legitimate to set random_order = True. 
        If not, set random_order = False. 
        '''
        x = np.array(x)
        assert all(x >=0),  'Negative value in a nonnegative population!'
        assert len(x) <= N, 'Sample size is larger than the population!'
        assert N > 0,       'Population size not positive!'
        assert N == int(N), 'Non-integer population size!'
        sample_total = 0.0
        mart = x[0]/t if t > 0 else 1
        mart_max = mart
        for j in range(1, len(x)):
            mart *= x[j]*(1-j/N)/(t - (1/N)*sample_total)
            if mart < 0:
                mart = np.inf
                break
            else:
                sample_total += x[j]
            mart_max = max(mart, mart_max)
        p = min((1/mart_max if random_order else 1/mart),1)
        return p 

    @classmethod
    def integral_from_roots(cls, c, maximal=True):
        '''
        Integrate the polynomial \prod_{k=1}^n (x-c_j) from 0 to 1, i.e.,
           \int_0^1 \prod_{k=1}^n (x-c_j) dx
        using a recursive algorithm devised by Steve Evans.
    
        If maximal == True, finds the maximum of the integrals over lower degrees:
           \max_{1 \le k \le n} \int_0^1 \prod_{j=1}^k (x-c_j) dx
    
        Parameters:
        -----------
        c : array of roots
    
        Returns
        ------
        the integral or maximum integral and the vector of nested integrals
        '''
        n = len(c)
        a = np.zeros((n+1,n+1))
        a[0,0]=1
        for k in np.arange(n):
            for j in np.arange(n+1):
                a[k+1,j] = -c[k]*((k+1-j)/(k+1))*a[k,j]
                a[k+1,j] += 0 if j==0 else (1-c[k])*(j/(k+1))*a[k,j-1]
        integrals = np.zeros(n)
        for k in np.arange(1,n+1):
            integrals[k-1] = np.sum(a[k,:])/(k+1)
        if maximal:
            integral = np.max(integrals[1:])
        else:
            integral = np.sum(a[n,:])/(n+1)
        return integral, integrals

    @classmethod
    def kaplan_martingale(cls, x, N, t=1/2, random_order = True):
        """
        p-value for the hypothesis that the mean of a nonnegative population with 
        N elements is t, based on a result of Kaplan, computed with a recursive 
        algorithm devised by Steve Evans.
    
        The alternative is that the mean is larger than t.
        If the random sample x is in the order in which the sample was drawn, it is
        legitimate to set random_order = True. 
        If not, set random_order = False. 
    
        If N = np.inf, treats the sampling as if it is with replacement.
        If N is finite, assumes the sample is drawn without replacement.
    
        Parameters:   
        ----------
        x : array-like
            the sample
        N : int
            population size. Use np.inf for sampling with replacement
        t : double
            the hypothesized population mean
        random_order : boolean
            is the sample in random order?
            
        Returns: 
        -------  
        p : double 
            p-value of the null
        mart_vec : array
            martingale as elements are added to the sample
          
        """
        x = np.array(x)
        assert all(x >=0),  'Negative value in a nonnegative population!'
        assert len(x) <= N, 'Sample size is larger than the population!'
        assert N > 0,       'Population size not positive!'
        if np.isfinite(N):
            assert N == int(N), 'Non-integer population size!'
        Stilde = (np.insert(np.cumsum(x),0,0)/N)[0:len(x)] # \tilde{S}_{j-1}
        t_minus_Stilde = t - Stilde
        mart_max = 1
        mart_vec = np.ones_like(x, dtype=np.float)
        if any(t_minus_Stilde < 0): # sample total exceeds hypothesized population total
            mart_max = np.inf
        else: 
            jtilde = 1 - np.array(list(range(len(x))))/N
            c = np.multiply(x, np.divide(jtilde, t_minus_Stilde))-1 
            r = -np.array([1/cc for cc in c[0:len(x)+1] if cc != 0]) # roots
            Y_norm = np.cumprod(np.array([cc for cc in c[0:len(x)+1] if cc != 0])) # mult constant
            integral, integrals = integral_from_roots(r, maximal = False)
            mart_vec = np.multiply(Y_norm,integrals)
            mart_max = max(mart_vec) if random_order else mart_vec[-1]
        p = min(1/mart_max,1)
        return p, mart_vec
    
    
# utilities
       
def check_audit_parameters(risk_function, g, error_rates, contests):
    """
    Check whether the audit parameters are valid; complain if not.
    
    Parameters:
    ---------
    risk_function : string
        the risk-measuring function for the audit
    g : double in [0, 1)
        padding for Kaplan-Markov or Kaplan-Wald 
        
    error_rates : dict
        expected rates of overstatement and understatement errors
        
    contests : dict of dicts 
        contest-specific information for the audit
        
    Returns:
    --------
    """
    if risk_function in ['kaplan_markov','kaplan_wald']:
        assert g >=0, 'g must be at least 0'
        assert g < 1, 'g must be less than 1'
    for r in ['o1_rate','o2_rate','u1_rate','u2_rate']:
        assert error_rates[r] >= 0, 'expected error rates must be nonnegative'
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

def write_audit_parameters(log_file, seed, replacement, risk_function, g, N_ballots, error_rates, contests):
    """
    Write audit parameters to log_file as a json structure
    
    Parameters:
    ---------
    log_file : string
        filename to write to
        
    seed : string
        seed for the PRNG for sampling ballots
    
    risk_function : string
        risk-measuring function used in the audit
        
    g : double
        padding for Kaplan-Wald and Kaplan-Markov
        
    error_rates : dict
        expected rates of overstatement and understatement errors
        
    contests : dict of dicts 
        contest-specific information for the audit
        
    Returns:
    --------
    """
    out = {"seed" : seed,
           "replacement" : replacement,
           "risk_function" : risk_function,
           "g" : g,
           "N_ballots" : N_ballots,
           "error_rates" : error_rates,
           "contests" : contests
          }
    with open(log_file, 'w') as f:
        json.dump(out, f)



# Unit tests

def test_make_plurality_assertions():
    winners = ["Alice","Bob"]
    losers = ["Candy","Dan"]
    asrtns = Assertion.make_plurality_assertions(winners, losers)
    assert asrtns['Alice v Candy'].assorter.assort({"Alice": 1}) == 1
    assert asrtns['Alice v Candy'].assorter.assort({"Bob": 1}) == 1/2
    assert asrtns['Alice v Candy'].assorter.assort({"Candy": 1}) == 0
    assert asrtns['Alice v Candy'].assorter.assort({"Dan": 1}) == 1/2

    assert asrtns['Alice v Dan'].assorter.assort({"Alice": 1}) == 1
    assert asrtns['Alice v Dan'].assorter.assort({"Bob": 1}) == 1/2
    assert asrtns['Alice v Dan'].assorter.assort({"Candy": 1}) == 1/2
    assert asrtns['Alice v Dan'].assorter.assort({"Dan": 1}) == 0
    
    assert asrtns['Bob v Candy'].assorter.assort({"Alice": 1}) == 1/2
    assert asrtns['Bob v Candy'].assorter.assort({"Bob": 1}) == 1
    assert asrtns['Bob v Candy'].assorter.assort({"Candy": 1}) == 0
    assert asrtns['Bob v Candy'].assorter.assort({"Dan": 1}) == 1/2

    assert asrtns['Bob v Dan'].assorter.assort({"Alice": 1}) == 1/2
    assert asrtns['Bob v Dan'].assorter.assort({"Bob": 1}) == 1
    assert asrtns['Bob v Dan'].assorter.assort({"Candy": 1}) == 1/2
    assert asrtns['Bob v Dan'].assorter.assort({"Dan": 1}) == 0

def test_supermajority_assorter():
    losers = ["Bob","Candy"]
    share_to_win = 2/3
    assn = Assertion.make_supermajority_assertion("AvBC","Alice", losers, share_to_win)

    votes = {"ID": 1, "votes": {"AvBC": {"Alice": 1}}}
    assert assn['Alice v all'].assorter.assort(votes) == 3/4, "wrong value for vote for winner"
    
    votes = {"Bob": True}
    assert assn['Alice v all'].assorter.assort(votes) == 0, "wrong value for vote for loser"
    
    votes = {"Dan": True}
    assert assn['Alice v all'].assorter.assort(votes) == 1/2, "wrong value for invalid vote--Dan"

    votes = {"Alice": True, "Bob": True}
    assert assn['Alice v all'].assorter.assort(votes) == 1/2, "wrong value for invalid vote--Alice & Bob"

    votes = {"Alice": False, "Bob": True, "Candy": True}
    assert assn['Alice v all'].assorter.assort(votes) == 1/2, "wrong value for invalid vote--Bob & Candy"


def test_rcv_lfunc_wo():
    votes = {"Alice": 1, "Bob": 2, "Candy": 3, "Dan": ''}
    assert CVR.rcv_lfunc_wo("Bob", "Alice", votes) == 1
    assert CVR.rcv_lfunc_wo("Alice", "Candy", votes) == 0
    assert CVR.rcv_lfunc_wo("Dan", "Candy", votes) == 1

def test_rcv_votefor_cand():
    votes = {"Alice": 1, "Bob": 2, "Candy": 3, "Dan": '', "Ross" : 4, "Aaron" : 5}
    remaining = ["Bob","Dan","Aaron","Candy"]
    assert CVR.rcv_votefor_cand("Candy", remaining, votes) == 0 
    assert CVR.rcv_votefor_cand("Alice", remaining, votes) == 0 
    assert CVR.rcv_votefor_cand("Bob", remaining, votes) == 1 
    assert CVR.rcv_votefor_cand("Aaron", remaining, votes) == 0

    remaining = ["Dan","Aaron","Candy"]
    assert CVR.rcv_votefor_cand("Candy", remaining, votes) == 1 
    assert CVR.rcv_votefor_cand("Alice", remaining, votes) == 0 
    assert CVR.rcv_votefor_cand("Bob", remaining, votes) == 0 
    assert CVR.rcv_votefor_cand("Aaron", remaining, votes) == 0

def test_rcv_assorter():
    import json
    with open('Data/334_361_vbm.json') as fid:
        data = json.load(fid)

        assertions = {}
        for audit in data['audits']:
            cands = [audit['winner']]
            for elim in audit['eliminated']:
                cands.append(elim)

            all_assertions = Assertion.make_assertions_from_json(cands, audit['assertions'])

            assertions[audit['contest']] = all_assertions
            
        assorter = assertions['334']['5 v 47'].assorter
        votes = {'5' : 1, '47' : 2}
        assert(assorter.assort(votes) == 1)

        votes = {'47' : 1, '5' : 2}
        assert(assorter.assort(votes) == 0)

        votes = {'3' : 1, '6' : 2}
        assert(assorter.assort(votes) == 0.5)

        votes = {'3' : 1, '47' : 2}
        assert(assorter.assort(votes) == 0)

        votes = {'3' : 1, '5' : 2}
        assert(assorter.assort(votes) == 0.5)

        assorter = assertions['334']['5 v 3 elim 1 6 47'].assorter
        votes = {'5' : 1, '47' : 2}
        assert(assorter.assort(votes) == 1)

        votes = {'47' : 1, '5' : 2}
        assert(assorter.assort(votes) == 1)

        votes = {'6' : 1, '1' : 2, '3' : 3, '5' : 4}
        assert(assorter.assort(votes) == 0)

        votes = {'3' : 1, '47' : 2}
        assert(assorter.assort(votes) == 0)

        votes = {}
        assert(assorter.assort(votes) == 0.5)

        votes = {'6' : 1, '47' : 2}
        assert(assorter.assort(votes) == 0.5)

        votes = {'6' : 1, '47' : 2, '5' : 3}
        assert(assorter.assort(votes) == 1)

        assorter = assertions['361']['28 v 50'].assorter
        votes = {'28' : 1, '50' : 2}
        assert(assorter.assort(votes) == 1)
        votes = {'28' : 1}
        assert(assorter.assort(votes) == 1)
        votes = {'50' : 1}
        assert(assorter.assort(votes) == 0)

        votes = {'27' : 1, '28' : 2}
        assert(assorter.assort(votes) == 0.5)

        votes = {'50' : 1, '28' : 2}
        assert(assorter.assort(votes) == 0)

        votes = {'27' : 1, '26' : 2}
        assert(assorter.assort(votes) == 0.5)

        votes = {}
        assert(assorter.assort(votes) == 0.5)

        assorter = assertions['361']['27 v 26 elim 28 50'].assorter
        votes = {'27' : 1}
        assert(assorter.assort(votes) == 1)

        votes = {'50' : 1, '27' : 2}
        assert(assorter.assort(votes) == 1)

        votes = {'28' : 1, '50' : 2, '27' : 3}
        assert(assorter.assort(votes) == 1)

        votes = {'28' : 1, '27' : 2, '50' : 3}
        assert(assorter.assort(votes) == 1)

        votes = {'26' : 1}
        assert(assorter.assort(votes) == 0)

        votes = {'50' : 1, '26' : 2}
        assert(assorter.assort(votes) == 0)

        votes = {'28' : 1, '50' : 2, '26' : 3}
        assert(assorter.assort(votes) == 0)

        votes = {'28' : 1, '26' : 2, '50' : 3}
        assert(assorter.assort(votes) == 0)

        votes = {'50' : 1}
        assert(assorter.assort(votes) == 0.5)
        votes = {}
        assert(assorter.assort(votes) == 0.5)

        votes = {'50' : 1, '28' : 2}
        assert(assorter.assort(votes) == 0.5)

        votes = {'28' : 1, '50' : 2}
        assert(assorter.assort(votes) == 0.5)

def test_kaplan_markov():
    s = np.ones(5)
    np.testing.assert_almost_equal(TestNonnegMean.kaplan_markov(s), 2**-5)
    s = np.array([1, 1, 1, 1, 1, 0])
    np.testing.assert_almost_equal(TestNonnegMean.kaplan_markov(s, g=0.1),(1.1/.6)**-5)
    np.testing.assert_almost_equal(TestNonnegMean.kaplan_markov(s, g=0.1, random_order = False),(1.1/.6)**-5 * .6/.1)
    s = np.array([1, -1])
    try:
        TestNonnegMean.kaplan_markov(s)
    except ValueError:
        pass
    else:
        raise AssertionError

def test_kaplan_wald():
    s = np.ones(5)
    np.testing.assert_almost_equal(TestNonnegMean.kaplan_wald(s), 2**-5)
    s = np.array([1, 1, 1, 1, 1, 0])
    np.testing.assert_almost_equal(TestNonnegMean.kaplan_wald(s, g=0.1), (1.9)**-5)
    np.testing.assert_almost_equal(TestNonnegMean.kaplan_wald(s, g=0.1, random_order = False),(1.9)**-5 * 10)
    s = np.array([1, -1])
    try:
        TestNonnegMean.kaplan_wald(s)
    except ValueError:
        pass
    else:
        raise AssertionError

def test_cvr_mean():
    pass  # FIX ME


if __name__ == "__main__":
    test_make_plurality_assertions()
    test_supermajority_assorter()
    test_rcv_lfunc_wo()
    test_rcv_votefor_cand()    
    test_rcv_assorter()
    test_kaplan_markov()
    test_kaplan_wald()
