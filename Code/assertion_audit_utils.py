import math
import numpy as np
import scipy as sp
import pandas as pd
import json
import csv
import warnings
from numpy import testing
from collections import OrderedDict
from cryptorandom.cryptorandom import SHA256, random
from cryptorandom.sample import random_permutation
from cryptorandom.sample import sample_by_index


class Assertion:
    '''
    Objects and methods for assertions about elections.
    
    An _assertion_ is a statement of the form 
      "the average value of this assorter applied to the ballots is greater than 1/2"
    The _assorter_ maps votes to nonnegative numbers not exceeding `upper_bound`
    '''
    
    JSON_ASSERTION_TYPES = ["WINNER_ONLY", "IRV_ELIMINATION"]  # supported json assertion types
    MANIFEST_TYPES = ["ALL","STYLE"]  # supported manifest types
    
    def __init__(self, contest = None, assorter = None, margin = 0, p_value = 1, proved = False):
        '''
        The assorter is callable; should produce a non-negative real.
        '''
        self.assorter = assorter
        self.contest = contest
        self.margin = margin
        self.p_value = p_value
        self.proved = proved
        
    def __str__(self):
        return "contest: " + str(self.contest) + " margin: " + self.margin \
               + " p-value: " + str(self.p_value) + " proved: " + str(self.proved) 

    def set_assorter(self, assorter):
        self.assorter = assorter
        
    def get_assorter(self):
        return self.assorter

    def set_contest(self, contest):
        self.contest = contest
        
    def get_contest(self):
        return self.contest

    def set_p_value(self, p_value):
        self.p_value = p_value
        
    def get_p_value(self):
        return self.p_value

    def set_margin(self, margin):
        self.margin = margin
        
    def get_margin(self):
        return self.margin

    def set_proved(self, proved):
        self.proved = proved
        
    def get_proved(self):
        return self.proved

    def assort(self, cvr):
        return self.assorter(cvr)
    
    def assorter_mean(self, cvr_list):
        '''
        find the mean of the assorter applied to a list of CVRs  
        
        Parameters:
        ----------
        cvr_list : list
            a list of cast-vote records
        
        Returns:
        ----------
        mean : float
            the mean value of the assorter over the list of cvrs
        '''
        return np.mean([self.assorter.assort(c) for c in cvr_list])      
        
    def assorter_sum(self, cvr_list):
        '''
        find the sum of the assorter applied to a list of CVRs   
        
        Parameters:
        ----------
        cvr_list : list of CVRs
            a list of cast-vote records
        
        Returns:
        ----------
        sum : float
            sum of the value of the assorter over a list of CVRs
        '''
        return np.sum([self.assorter.assort(c) for c in cvr_list])

    def assorter_margin(self, cvr_list):
        '''
        find the margin for a list of Cvrs. 
        By definition, the margin is the mean of the assorter minus 1/2, times 2
        
        Parameters:
        ----------
        cvr_list : list
            a list of cast-vote records
        
        Returns:
        ----------
        margin : float
        '''
        return 2*self.assorter_mean(cvr_list)-1

    def overstatement(self, mvr, cvr, manifest_type="STYLE"):
        '''
        overstatement error for a CVR compared to the human reading of the ballot
        
        If manifest_type == "STYLE", then if the CVR contains the contest but the MVR does 
        not, that is considered to be an overstatement, because the ballot should have 
        contained the contest.
        
        If manifest_type == "ALL", then if the CVR contains the contest but the MVR does not,
        the MVR is considered to be a non-vote in the contest.
        
        Phantom CVRs and MVRs are treated specially:
            A phantom CVR is considered a non-vote in every contest (assort() = 1/2).
            A phantom MVR is considered a vote for the loser (i.e., assort() = 0) in every 
            contest.

        Parameters:
        -----------
        mvr : Cvr
            the manual interpretation of voter intent
        cvr : Cvr
            the machine-reported cast vote record
            
        Returns:
        --------
        overstatement : float
            the overstatement error
        '''        
        assert manifest_type in Assertion.MANIFEST_TYPES, "unrecognized manifest type"
        if manifest_type == "ALL":
            overstatement = self.assorter.assort(cvr)\
                            - (self.assorter.assort(mvr) if not mvr.phantom else 0)
        elif manifest_type == "STYLE":
            if mvr.has_contest(self.contest):
                overstatement = self.assorter.assort(cvr)-self.assorter.assort(mvr)
            elif not mvr.has_contest(self.contest) and cvr.has_contest(self.contest):
                overstatement = self.assorter.assort(cvr)
            elif mvr.phantom and cvr.phantom:
                overstatement = 1/2
            else:
                overstatement = 0
        else:
            raise NotImplementedError
        return overstatement   
    
    def overstatement_assorter(self, mvr, cvr, margin, manifest_type="STYLE"):
        '''
        assorter that corresponds to normalized overstatement error for an assertion
        
        If manifest_type == "STYLE", then if the CVR contains the contest but the MVR does not,
        that is considered to be an overstatement, because the ballot is presumed to contain
        the contest.
        
        If manifest_type == "ALL", then if the CVR contains the contest but the MVR does not,
        the MVR is considered to be a non-vote in the contest.

        Parameters:
        -----------
        mvr : Cvr
            the manual interpretation of voter intent
        cvr : Cvr
            the machine-reported cast vote record
        margin : float
            2*(assorter applied to all CVRs) - 1, the assorter margin
            
        Returns:
        --------
        over : float
            (1-o/u)/(2-v/u), where 
                o is the overstatement
                u is the upper bound on the value the assorter assigns to any ballot
                v is the assorter margin
        '''        
        return (1-self.overstatement(mvr, cvr, manifest_type)/self.assorter.upper_bound)\
                /(2-margin/self.assorter.upper_bound)
        
      
    @classmethod
    def make_plurality_assertions(cls, contest, winners, losers):
        '''
        Construct assertions that imply the winner(s) got more votes than the loser(s).
        
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
        contest : string
            identifier of contest to which the assertion applies
        winner : 
            identifier of winning candidate
        losers : list
            list of identifiers of losing candidate(s)
        share_to_win : float
            fraction of the valid votes the winner must get to win        
        
        Returns:
        --------
        a dict containing one Assertion
        
        '''
        assert share_to_win < 1, "share_to_win must be less than 1"

        assertions = {}
        wl_pair = winner + ' v all'
        cands = losers.copy()
        cands.append(winner)
        assertions[wl_pair] = Assertion(contest, \
                                 Assorter(contest=contest, assort = lambda c, contest=contest: \
                                 CVR.as_vote(CVR.get_vote_from_cvr(contest, winner, \
                                 c))/(2*share_to_win) \
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
        contest : string
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
                winner_func = lambda v, contest=contest, winr=winr : 1 \
                              if CVR.get_vote_from_cvr(contest, winr, v) == 1 else 0

                # CVR is a vote for the loser if they appear and the 
                # winner does not, or they appear before the winner
                loser_func = lambda v, contest=contest, winr=winr, losr=losr : \
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
        '''
        Construct all the assertions to audit the contests.
        
        Parameters:
        -----------
        contests : dict
            the contest-level data
        
        Returns:
        --------
        A dict of dicts of Assertion objects
        '''
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
    '''
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
        maps dict of selections into float
    
    upper_bound : float
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
        assort : callable
            maps a dict of votes into [0, \infty)
        winner : callable
            maps a pattern into {0, 1}
        loser  : callable
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
    '''
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
            {"id": "A-001-01", "votes": {"mayor": {"Alice": True}}}
            {"id": "A-001-01", "votes": {"mayor": {"Alice": "marked"}}}
            {"id": "A-001-01", "votes": {"mayor": {"Alice": 5}}}
            {"id": "A-001-01", "votes": {"mayor": {"Alice": 1, "Bob": 0, "Candy": 0, "Dan": ""}}}
            {"id": "A-001-01", "votes": {"mayor": {"Alice": True, "Bob": False}}}
    A CVR that contains a vote for Alice for "mayor" and a vote for Bob for "DA" could be represented as
            {"id": "A-001-01", "votes": {"mayor": {"Alice": True}, "DA": {"Bob": True}}}

    NOTE: some methods distinguish between a CVR that contains a particular contest, but no valid
    vote in that contest, and a CVR that does not contain that contest at all. Thus, the following
    are not equivalent:
            {"id": "A-001-01", "votes": {"mayor": {}} }
            and
            {"id": "A-001-01", "votes": {} }
                    
    Ranked votes also have simple representation, e.g., if the CVR is
            {"id": "A-001-01", "votes": {"mayor": {"Alice": 1, "Bob": 2, "Candy": 3, "Dan": ''}}}
    Then int(vote_for("Candy","mayor"))=3, Candy's rank in the "mayor" contest.
    
    CVRs can be flagged as "phantoms" to account for cards not listed in the manifest.
     
    Methods:
    --------
    
    get_votefor :  
         get_votefor(candidate, contest, votes) returns the value in the votes dict for the key `candidate`, or
         False if the candidate did not get a vote or the contest is not in the CVR
    set_votes :  
         updates the votes dict; overrides previous votes and/or creates votes for additional contests or candidates
    get_votes : returns complete votes dict for a contest
    get_id : returns ballot id
    set_id : updates the ballot id
        
    '''
    
    def __init__(self, id = None, votes = {}, phantom = False):
        self.votes = votes
        self.id = id
        self.phantom = phantom
        
    def __str__(self):
        return "id: " + str(self.id) + " votes: " + str(self.votes) + " phantom: " + str(self.phantom)
        
    def get_votes(self):
        return self.votes
    
    def set_votes(self, votes):
        self.votes.update(votes)
            
    def get_id(self):
        return self.id
    
    def set_id(self, id):
        self.id = id
        
    def is_phantom(self):
        return self.phantom
    
    def set_phantom(self, phantom):
        self.phantom = phantom
    
    def get_votefor(self, contest, candidate):
        return CVR.get_vote_from_cvr(contest, candidate, self)
    
    def has_contest(self, contest):
        return contest in self.votes
    
    @classmethod
    def cvrs_to_json(cls, cvr):
        return json.dumps(cvr)
    
    @classmethod
    def from_dict(cls, cvr_dict):
        '''
        Construct a list of CVR objects from a list of dicts containing cvr data
        
        Parameters:
        -----------
        cvr_dict: a list of dicts, one per cvr
        
        Returns:
        ---------
        list of CVR objects
        '''
        cvr_list = []
        for c in cvr_dict:
            phantom = False if 'phantom' not in c.keys() else c['phantom']
            cvr_list.append(CVR(id = c['id'], votes = c['votes'], phantom=phantom))
        return cvr_list
    
    @classmethod
    def from_raire(cls, raire, phantom=False):
        '''
        Create a list of CVR objects from a list of cvrs in RAIRE format
        
        Parameters:
        -----------
        raire : list of comma-separated values
            source in RAIRE format. From the RAIRE documentation:
            The RAIRE format (for later processing) is a CSV file.
            First line: number of contests.
            Next, a line for each contest
             Contest,id,N,C1,C2,C3 ...
                id is the contest id
                N is the number of candidates in that contest
                and C1, ... are the candidate id's relevant to that contest.
            Then a line for every ranking that appears on a ballot:
             Contest id,Ballot id,R1,R2,R3,...
            where the Ri's are the unique candidate ids.
            
            The CVR file is assumed to have been read using csv.reader(), so each row has
            been split.
            
        Returns:
        --------
        list of CVR objects corresponding to the RAIRE cvrs
        '''
        skip = int(raire[0][0])
        cvr_list = []
        for c in raire[(skip+1):]:
            contest = c[0]
            id = c[1]
            votes = {}
            for j in range(2, len(c)):
                votes[str(c[j])] = j-1
            cvr_list.append(CVR.from_vote(votes, id=id, contest=contest, phantom=phantom))
        return CVR.merge_cvrs(cvr_list)
    
    @classmethod
    def merge_cvrs(cls, cvr_list):
        '''
        Takes a list of CVRs that might contain duplicated ballot ids and merges the votes
        so that each identifier is listed only once, and votes from different records for that
        identifier are merged.
        The merge is in the order of the list: if a later mention of a ballot id has votes 
        for the same contest as a previous mention, the votes in that contest are updated
        per the later mention.
        
        If any of the CVRs has phantom==False, sets phantom==False in the result.
        
        
        Parameters:
        -----------
        cvr_list : list of CVRs
        
        Returns:
        -----------
        list of merged CVRs
        '''
        od = OrderedDict()
        for c in cvr_list:
            if c.id not in od:
                od[c.id] = c
            else:
                od[c.id].set_votes(c.votes)
                od[c.id].set_phantom(c.phantom and od[c.id].phantom)
        return [v for v in od.values()]
    
    @classmethod
    def from_vote(cls, vote, id = 1, contest = 'AvB', phantom=False):
        '''
        Wraps a vote and creates a CVR, for unit tests
    
        Parameters:
        ----------
        vote : dict of votes in one contest
    
        Returns:
        --------
        CVR containing that vote in the contest "AvB", with CVR id=1.
        '''
        return CVR(id=id, votes={contest: vote}, phantom=phantom)

    @classmethod
    def as_vote(cls, v):
        return int(bool(v))
    
    @classmethod
    def as_rank(cls, v):
        return int(v)
    
    @classmethod
    def get_vote_from_votes(cls, contest, candidate, votes):
        '''
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
        '''
        return False if (contest not in votes or candidate not in votes[contest])\
               else votes[contest][candidate]
    
    @classmethod
    def get_vote_from_cvr(cls, contest, candidate, cvr):
        '''
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
        '''
        return False if (contest not in cvr.votes or candidate not in cvr.votes[contest])\
               else cvr.votes[contest][candidate]
    
    @classmethod
    def has_one_vote(cls, contest, candidates, cvr):
        '''
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
        '''
        v = np.sum([0 if c not in cvr.votes[contest] else bool(cvr.votes[contest][c]) \
                    for c in candidates])
        return True if v==1 else False
    
    @classmethod
    def rcv_lfunc_wo(cls, contest, winner, loser, cvr):
        '''
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

        cvr : a CVR object

        Returns:
        --------
        1 if the given vote is a vote for 'loser' and 0 otherwise
        '''
        rank_winner = CVR.get_vote_from_cvr(contest, winner, cvr)
        rank_loser = CVR.get_vote_from_cvr(contest, loser, cvr)

        if not bool(rank_winner) and bool(rank_loser):
            return 1
        elif bool(rank_winner) and bool(rank_loser) and rank_loser < rank_winner:
            return 1
        else:
            return 0 

    @classmethod
    def rcv_votefor_cand(cls, contest, cand, remaining, cvr):
        '''
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
        '''
        if not cand in remaining:
            return 0

        rank_cand = CVR.get_vote_from_cvr(contest, cand, cvr)

        if not bool(rank_cand):
            return 0
        else:
            for altc in remaining:
                if altc == cand:
                    continue
                rank_altc = CVR.get_vote_from_cvr(contest, altc, cvr)
                if bool(rank_altc) and rank_altc <= rank_cand:
                    return 0
            return 1 

class TestNonnegMean:
    '''
    Tests of the hypothesis that the mean of a non-negative population is less than t.
        Several tests are implemented, all ultimately based on martingales:
            Kaplan-Markov
            Kaplan-Wald
            Kaplan-Kolmogorov
            Wald SPRT with replacement (only for binary-valued populations)
            Wald SPRT without replacement (only for binary-valued populations)
            Kaplan's martingale (KMart)        
    '''
    
    TESTS = ['kaplan_markov','kaplan_wald','kaplan_kolmogorov','wald_sprt','kaplan_martingale']
    
    @classmethod
    def wald_sprt(cls, x, N, t = 1/2, p1=1, random_order = True):
        '''
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
        t : float in (0,1)
            hypothesized fraction of ones in the population
        p1 : float in (0,1) greater than t
            alternative hypothesized fraction of ones in the population
        random_order : Boolean
            if the data are in random order, setting this to True can improve the power.
            If the data are not in random order, set to False
        '''
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
        return 1/np.max(np.cumprod(terms)) if random_order else 1/np.prod(terms)

    @classmethod
    def kaplan_markov(cls, x, t=1/2, g=0, random_order=True):
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
        x : array-like
            the sample
        t : float
            the null value of the mean
        g : float
            "padding" in case there any values in the sample are zero
        random_order : Boolean
            if the sample is in random order, it is legitimate to stop early, which 
            can yield a more powerful test. See above.
        
        Returns:
        --------
        p-value
        
        '''       
        if any(x < 0):
            raise ValueError('Negative value in sample from a nonnegative population.')
        return np.min([1, np.min(np.cumprod((t+g)/(x+g))) if random_order else np.prod((t+g)/(x+g))])

    @classmethod
    def kaplan_wald(cls, x, t=1/2, g=0, random_order=True):
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
        x : array-like
            the sample
        t : float
            the null value of the mean
        g : float
            "padding" in case there any values in the sample are zero
        random_order : Boolean
            if the sample is in random order, it is legitimate to stop early, which 
            can yield a more powerful test. See above.

        Returns:
        --------
        p-value
       
        '''       
        if g < 0:
            raise ValueError('g cannot be negative')
        if any(x < 0):
            raise ValueError('Negative value in sample from a nonnegative population.')
        return np.min([1, 1/np.max(np.cumprod((1-g)*x/t + g)) if random_order \
                       else 1/np.prod((1-g)*x/t + g)])
    
    @classmethod
    def kaplan_kolmogorov(cls, x, N, t=1/2, g=0, random_order = True):
        '''
        p-value for the hypothesis that the mean of a nonnegative population with N
        elements is t. The alternative is that the mean is less than t.
        If the random sample x is in the order in which the sample was drawn, it is
        legitimate to set random_order = True. 
        If not, set random_order = False. 
        
        g is a tuning parameter to protect against data values equal to zero.
        g should be in [0, 1)
        
        Parameters:
        -----------
        x : list
            observations
        N : int
            population size
        t : float
            null value of the population mean
        g : float in [0, 1)
            "padding" to protect against zeros
        '''
        x = np.array(x)
        assert all(x >=0),  'Negative value in a nonnegative population!'
        assert len(x) <= N, 'Sample size is larger than the population!'
        assert N > 0,       'Population size not positive!'
        assert N == int(N), 'Non-integer population size!'
        
        sample_total = 0.0
        mart = (x[0]+g)/(t+g) if t > 0 else 1
        mart_max = mart
        for j in range(1, len(x)):
            mart *= (x[j]+g)*(1-j/N)/(t+g - (1/N)*sample_total)
            if mart < 0:
                mart = np.inf
                break
            else:
                sample_total += x[j]+g
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
        '''
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
        t : float
            the hypothesized population mean
        random_order : boolean
            is the sample in random order?
            
        Returns: 
        -------  
        p : float 
            p-value of the null
        mart_vec : array
            martingale as elements are added to the sample
          
        '''
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

        jtilde = 1 - np.array(list(range(len(x))))/N
        # The term being mutliplied by gamma in the SHANGRLA paper
        c = x * jtilde / t_minus_Stilde - 1
        # as a convention, 0/0 is 1 here. This prevents c from
        # having nans and ensures the martingale property
        c[np.logical_and(x == 0, t_minus_Stilde == 0)] = 0 
        c_nonzero = c[c != 0]
        # integral should not contain c_j if c_j = 0
        r = -np.array(1/c_nonzero) # roots
        # Create a copy of c but 0s replaced with 1s
        # (for computing Y_norm)
        c_zeroone = c.copy()
        c_zeroone[c == 0] = 1
        Y_norm = np.cumprod(np.array(c_zeroone)) # mult constant
        integral, integrals = TestNonnegMean.integral_from_roots(r, maximal = False)
        # Create integrals vector of same length as c
        integrals_all = np.ones(len(c))
        # Whenever c was not zero (i.e. 1/c could be passed to
        # integrals_from_roots), set integrals_all to the returned
        # value

        integrals_all[c != 0] = integrals
        mart_vec = Y_norm * integrals_all
        mart_max = max(mart_vec)

        # If t - Stilde < 0, then we know deterministically
        # that the null is false.
        mart_vec[t_minus_Stilde < 0] = np.inf

        # get p-value as inverse of martingale
        p = min(1/mart_max,1)

        return p, mart_vec                  
        
    @classmethod
    def initial_sample_size(cls, risk_function, N, margin, error_rate, alpha=0.05, t=1/2, reps=None,\
                            bias_up = True, quantile=0.5, seed=1234567890):
        '''
        Estimate the sample size needed to reject the null hypothesis that the population 
        mean is <=t at significance level alpha, for the specified risk function, on the 
        assumption that the rate of one-vote overstatements is error_rate.  This function is
        for a single contest.
        
        This function is [currently] only for comparison audits, not ballot-polling audits.
        
        Implements two strategies:

            1. If reps is not None, the function uses simulations to estimate the <quantile> quantile
            of sample size required. The simulations use the numpy Mersenne Twister PRNG.
            
            2. If reps is None, puts discrepancies into the sample in a deterministic way, starting 
            with a discrepancy in the first item if bias_up is true, then including an additional discrepancy after 
            every int(1/error_rate) items in the sample. "Frontloading" the errors (bias_up == True) should make this
            _slightly_ conservative on average
        
        
        Parameters:
        -----------
        risk_function : callable
            risk function to use. risk_function should take one argument, x. 
        N : int
            population size, or N = np.infty for sampling with replacement
        margin : float
            assorter margin 
        error_rate : float 
            assumed rate of 1-vote overstatements 
        alpha : float
            significance level in (0, 0.5)
        t : float
            hypothesized value of the population mean
        reps : int
            if reps is not none, performs reps simulations to estimate the <quantile> quantile
            of sample sizes
        bias_up : boolean
            if True, front loads the discrepancies (biases sample size up). 
            If False, back loads them (biases sample size down).
        quantile : float
            quantile of the distribution of sample sizes to report, if reps is not None.
            If reps is None, quantile is not used
        seed : int
            if reps is not none, use this value as the seed for simulations.
            
        Returns:
        --------
        sample_size : int
            sample size estimated to be sufficient to confirm the outcome if one-vote overstatements 
            are not more frequent than the assumed rate
            
        '''
                             
        assert alpha > 0 and alpha < 1/2
        assert margin > 0
        clean = 1/(2-margin)
        one_vote_over = 0.5/(2-margin)
        if reps is None:
            offset = 0 if bias_up else 1                
            p = 1
            j = 0
            while (p > alpha) and (j <= N):
                j += 1
                x = clean*np.ones(j)
                for k in range(j):
                    x[k] = one_vote_over if (k+offset) % int(1/error_rate) == 0 else x[k]                   
                p = risk_function(x)
            sam_size = j
        else:
            prng = np.random.RandomState(1234567890)  # use the Mersenne Twister for speed
            sams = np.zeros(int(reps))
            for r in range(reps):
                new_size = 0
                pop = clean*np.ones(N)
                inx = (prng.random(size=N) <= error_rate)  # randomly allocate errors
                pop[inx] = one_vote_over
                j = 0
                p = 1
                while (p > alpha) and (j <= N):
                    j += 1
                    p = risk_function(pop[:j])
                sams[r] = j
            sam_size = np.quantile(sams, quantile)
        return sam_size
          
# utilities
       
def check_audit_parameters(risk_function, g, error_rate, contests):
    '''
    Check whether the audit parameters are valid; complain if not.
    
    Parameters:
    ---------
    risk_function : string
        the risk-measuring function for the audit
    g : float in [0, 1)
        padding for Kaplan-Markov or Kaplan-Wald 
        
    error_rate : float
        expected rate of 1-vote overstatements
        
    contests : dict of dicts 
        contest-specific information for the audit
        
    Returns:
    --------
    '''
    if risk_function in ['kaplan_markov','kaplan_wald']:
        assert g >=0, 'g must be at least 0'
        assert g < 1, 'g must be less than 1'
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

def find_margins(contests, assertions, cvr_list):
    '''
    Find all the assorter margins in a set of Assertions. Updates the dict of dicts of assertions,
    and the contest dict.
    
    This function is primarily about side-effects
    
    Parameters:
    -----------
    contests : dict of contest data
    assertions : dict of dicts of Assertions
        Keys in the main dict are contests; keys in the contained dicts are Assertions
    
    Returns:
    --------
    min_margin : float
        smallest margin in the audit        
    '''
    assorter_means = {}
    min_margin = np.infty
    for c in contests:
        contests[c]['margins'] = {}
        for asrtn in assertions[c]:
            # find mean of the assertion for the CVRs
            amean = assertions[c][asrtn].assorter_mean(cvr_list)
            if amean < 1/2:
                warn("assertion {} not satisfied by CVRs: mean value is {}".format(asrtn, amean))
            margin = 2*amean-1
            assertions[c][asrtn].margin = margin
            contests[c]['margins'].update({asrtn: margin})
            min_margin = np.min([min_margin, margin])
    return min_margin

def find_p_values(contests, assertions, mvr_sample, cvr_sample, manifest_type, risk_function):
    '''
    Find the p-value for every assertion in assertions; update data structure to
    include the p-values for the assertions, flag "proved" assertions, and note 
    the maximum p-value for each contest.
    
    Primarily about side-effects.
    
    Parameters:
    -----------
    contests : dict of dicts
        the contest data structure. outer keys are contest identifiers; inner keys are assertions
        
    assertions : dict of dicts of assertions
    
    mvr_sample : list of CVR objects
        the manually ascertained voter intent from sheets, including entries for phantoms
    
    cvr_sample : list of CVR objects
        the cvrs for the same sheets
        
    manifest_type : string
        "ALL" or "STYLE". See documentation
        
    risk_function : callable
        function to calculate the p-value from overstatement_assorter values
        
    Returns:
    --------
    p_max : float
        largest p-value for any assertion in any contest
        
    '''
    assert len(mvr_sample) == len(cvr_sample), "unequal numbers of cvrs and mvrs"
    p_max = 0
    for c in contests.keys():
        contests[c]['p_values'] = {}
        contests[c]['proved'] = {}
        contest_max_p = 0
        for asrtn in assertions[c]:
            a = assertions[c][asrtn]
            d = [a.overstatement_assorter(mvr_sample[i], cvr_sample[i],\
                 a.margin, manifest_type=manifest_type) for i in range(len(mvr_sample))]
            a.p_value = risk_function(d)
            a.proved = (a.p_value <= contests[c]['risk_limit']) or a.proved
            contests[c]['p_values'].update({asrtn: a.p_value})
            contests[c]['proved'].update({asrtn: int(a.proved)})
            contest_max_p = np.max([contest_max_p, a.p_value])
        contests[c].update({'max_p': contest_max_p})
        p_max = np.max([p_max, contests[c]['max_p']])
    return p_max

def find_sample_size(contests, assertions, sample_size_function):
    '''
    Find initial sample size
    
    Parameters:
    -----------
    contests : dict of dicts
    assertion : dict of dicts
    sample_size_function : callable
        takes two parameters, the margin and the risk limit; returns a sample size
    
    Returns:
    --------
    size : int
        sample size expected to be adequate to confirm all assertions
    '''
    sample_size = 0
    for c in contests:
        risk = contests[c]['risk_limit']
        for a in assertions[c]:
            margin = assertions[c][a].margin
            n = sample_size_function(margin, risk)
            sample_size = np.max([sample_size, n] )
    return sample_size

def prep_sample(mvr_sample, cvr_sample):
    '''
    prepare the MVRs and CVRs for comparison by putting the MVRs into the same (random) order
    in which the CVRs were selected
    
    conduct data integrity checks.
    
    Side-effects: sorts the mvr sample into the same order as the cvr sample
    
    Parameters:
    -----------
    mvr_sample: list of CVR objects 
        the manually determined votes for the audited cards
    cvr_sample: list of CVR objects
        the electronic vote record for the audited cards 
    
    Returns:
    --------
    '''
    id_lookup = [c.id for c in cvr_sample]
    mvr_sample.sort(key= lambda x: id_lookup.index(x.id))

    assert len(cvr_sample) == len(mvr_sample),\
        "Number of cvrs ({}) and mvrs ({}) doesn't match".format(len(cvr_sample), len(mvr_sample))
    for i in range(len(cvr_sample)):
        assert mvr_sample[i].id == cvr_sample[i].id, \
    "Mismatch between id of cvr ({}) and mvr ({})".format(cvr_sample[i].id, mvr_sample[i].id)

def new_sample_size(contests, assertions, mvr_sample, cvr_sample, manifest_type,\
                    risk_function, quantile=0.5, reps=200, seed=1234567890):
    '''
    Estimate the total sample size expected to allow the audit to complete,
    if discrepancies continue at the same rate already observed.
    
    Uses simulations. For speed, uses the numpy.random Mersenne Twister instead of cryptorandom.
        
    Parameters:
    -----------
    contests : dict of dicts
        the contest data structure. outer keys are contest identifiers; inner keys are assertions
        
    assertions : dict of dicts of assertions
    
    mvr_sample : list of CVR objects
        the manually ascertained voter intent from sheets, including entries for phantoms
    
    cvr_sample : list of CVR objects
        the cvrs for the same sheets
        
    manifest_type : string
        "ALL" or "STYLE". See documentation
        
    risk_function : callable
        function to calculate the p-value from overstatement_assorter values.
        Should take one argument, the sample x
        
    quantile : float
        estimated quantile of the sample size to return
        
    reps : int
        number of replications to use to estimate the quantile
    
    seed : int
        seed for the Mersenne Twister prng
    
    Returns:
    --------
    new_size : int
        new sample size
    sams : array of ints
        array of all sizes found in the simulation
    '''
    new_size = 0
    prng = np.random.RandomState(seed=seed)
    sams = np.zeros(reps)
    for r in range(reps):
        new_size = 0
        for c in contests:
            for asrtn in assertions[c]:
                if not assertions[c][asrtn].proved:    
                    a = assertions[c][asrtn]
                    p = a.p_value
                    d = [a.overstatement_assorter(mvr_sample[i], cvr_sample[i],\
                         a.margin, manifest_type=manifest_type) for i in range(len(mvr_sample))]
                    while p > contests[c]['risk_limit']:
                        one_more = sample_by_index(len(d), 1, prng=prng)[0]
                        d.append(d[one_more-1])
                        p = risk_function(d)
                    new_size = np.max([new_size, len(d)])
        sams[r] = new_size 
    new_size = np.quantile(sams, quantile)
    return new_size, sams

def summarize_status(contests, assertions):
    '''
    Determine whether the audit of individual assertions, contests, and the election
    are finished.
    
    Prints a summary.
    
    Parameters:
    -----------
    contests : dict of dicts
        dict of contest information
    assertions : dict of dicts
        the assertions
    
    
    Returns:
    --------
    done : boolean
        is the audit finished?'''
    done = True
    for c in contests:
        print("p-values for assertions in contest {}".format(c))
        cpmax = 0
        for a in assertions[c]:
            p = assertions[c][a].p_value
            cpmax = np.max([cpmax,p])
            print(a, p)
        if cpmax <= contests[c]['risk_limit']:
            print("\ncontest {} AUDIT COMPLETE at risk limit {}. Attained risk {}".format(\
                c, contests[c]['risk_limit'], cpmax))
        else:
            done = False
            print("\ncontest {} audit INCOMPLETE at risk limit {}. Attained risk {}".format(\
                c, contests[c]['risk_limit'], cpmax))
            print("assertions remaining to be proved:")
            for a in assertions[c]:
                if assertions[c][a].p_value > contests[c]['risk_limit']:
                    print("{}: current risk {}".format(a, assertions[c][a].p_value))
    return done

def write_audit_parameters(log_file, seed, replacement, risk_function, g, \
                           N_cards, n_cvrs, manifest_cards, phantom_cards, error_rate, contests):
    '''
    Write audit parameters to log_file as a json structure
    
    Parameters:
    ---------
    log_file : string
        filename to write to
        
    seed : string
        seed for the PRNG for sampling ballots
    
    risk_function : string
        risk-measuring function used in the audit
        
    g : float
        padding for Kaplan-Wald and Kaplan-Markov
        
    error_rate : float
        expected rate of 1-vote overstatements
        
    contests : dict of dicts 
        contest-specific information for the audit
        
    Returns:
    --------
    no return value
    '''                      
    out = {"seed" : seed,
           "replacement" : replacement,
           "risk_function" : risk_function,
           "g" : float(g),
           "N_cards" : int(N_cards),
           "n_cvrs" : int(n_cvrs),
           "manifest_cards" : int(manifest_cards),
           "phantom_cards" : int(phantom_cards),
           "error_rate" : error_rate,
           "contests" : contests
          }
    with open(log_file, 'w') as f:
        f.write(json.dumps(out))

def trim_ints(x):
    '''
    turn int64 into an int
    
    Parameters
    ----------
    x : int64
    
    Returns
    -------
    int(x) : int
   '''
    if isinstance(x, np.int64): 
        return int(x)  
    else:
        raise TypeError


# Unit tests

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

def test_rcv_lfunc_wo():
    votes = CVR.from_vote({"Alice": 1, "Bob": 2, "Candy": 3, "Dan": ''})
    assert CVR.rcv_lfunc_wo("AvB", "Bob", "Alice", votes) == 1
    assert CVR.rcv_lfunc_wo("AvB", "Alice", "Candy", votes) == 0
    assert CVR.rcv_lfunc_wo("AvB", "Dan", "Candy", votes) == 1

def test_rcv_votefor_cand():
    votes = CVR.from_vote({"Alice": 1, "Bob": 2, "Candy": 3, "Dan": '', "Ross" : 4, "Aaron" : 5})
    remaining = ["Bob","Dan","Aaron","Candy"]
    assert CVR.rcv_votefor_cand("AvB", "Candy", remaining, votes) == 0 
    assert CVR.rcv_votefor_cand("AvB", "Alice", remaining, votes) == 0 
    assert CVR.rcv_votefor_cand("AvB", "Bob", remaining, votes) == 1 
    assert CVR.rcv_votefor_cand("AvB", "Aaron", remaining, votes) == 0

    remaining = ["Dan","Aaron","Candy"]
    assert CVR.rcv_votefor_cand("AvB", "Candy", remaining, votes) == 1 
    assert CVR.rcv_votefor_cand("AvB", "Alice", remaining, votes) == 0 
    assert CVR.rcv_votefor_cand("AvB", "Bob", remaining, votes) == 0 
    assert CVR.rcv_votefor_cand("AvB", "Aaron", remaining, votes) == 0

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
        votes = CVR.from_vote({'5' : 1, '47' : 2})
        assert(assorter.assort(votes) == 1)

        votes = CVR.from_vote({'47' : 1, '5' : 2})
        assert(assorter.assort(votes) == 0)

        votes = CVR.from_vote({'3' : 1, '6' : 2})
        assert(assorter.assort(votes) == 0.5)

        votes = CVR.from_vote({'3' : 1, '47' : 2})
        assert(assorter.assort(votes) == 0)

        votes = CVR.from_vote({'3' : 1, '5' : 2})
        assert(assorter.assort(votes) == 0.5)

        assorter = assertions['334']['5 v 3 elim 1 6 47'].assorter
        votes = CVR.from_vote({'5' : 1, '47' : 2})
        assert(assorter.assort(votes) == 1)

        votes = CVR.from_vote({'47' : 1, '5' : 2})
        assert(assorter.assort(votes) == 1)

        votes = CVR.from_vote({'6' : 1, '1' : 2, '3' : 3, '5' : 4})
        assert(assorter.assort(votes) == 0)

        votes = CVR.from_vote({'3' : 1, '47' : 2})
        assert(assorter.assort(votes) == 0)

        votes = CVR.from_vote({})
        assert(assorter.assort(votes) == 0.5)

        votes = CVR.from_vote({'6' : 1, '47' : 2})
        assert(assorter.assort(votes) == 0.5)

        votes = CVR.from_vote({'6' : 1, '47' : 2, '5' : 3})
        assert(assorter.assort(votes) == 1)

        assorter = assertions['361']['28 v 50'].assorter
        votes = CVR.from_vote({'28' : 1, '50' : 2})
        assert(assorter.assort(votes) == 1)
        votes = CVR.from_vote({'28' : 1})
        assert(assorter.assort(votes) == 1)
        votes = CVR.from_vote({'50' : 1})
        assert(assorter.assort(votes) == 0)

        votes = CVR.from_vote({'27' : 1, '28' : 2})
        assert(assorter.assort(votes) == 0.5)

        votes = CVR.from_vote({'50' : 1, '28' : 2})
        assert(assorter.assort(votes) == 0)

        votes = CVR.from_vote({'27' : 1, '26' : 2})
        assert(assorter.assort(votes) == 0.5)

        votes = CVR.from_vote({})
        assert(assorter.assort(votes) == 0.5)

        assorter = assertions['361']['27 v 26 elim 28 50'].assorter
        votes = CVR.from_vote({'27' : 1})
        assert(assorter.assort(votes) == 1)

        votes = CVR.from_vote({'50' : 1, '27' : 2})
        assert(assorter.assort(votes) == 1)

        votes = CVR.from_vote({'28' : 1, '50' : 2, '27' : 3})
        assert(assorter.assort(votes) == 1)

        votes = CVR.from_vote({'28' : 1, '27' : 2, '50' : 3})
        assert(assorter.assort(votes) == 1)

        votes = CVR.from_vote({'26' : 1})
        assert(assorter.assort(votes) == 0)

        votes = CVR.from_vote({'50' : 1, '26' : 2})
        assert(assorter.assort(votes) == 0)

        votes = CVR.from_vote({'28' : 1, '50' : 2, '26' : 3})
        assert(assorter.assort(votes) == 0)

        votes = CVR.from_vote({'28' : 1, '26' : 2, '50' : 3})
        assert(assorter.assort(votes) == 0)

        votes = CVR.from_vote({'50' : 1})
        assert(assorter.assort(votes) == 0.5)
        votes = CVR.from_vote({})
        assert(assorter.assort(votes) == 0.5)

        votes = CVR.from_vote({'50' : 1, '28' : 2})
        assert(assorter.assort(votes) == 0.5)

        votes = CVR.from_vote({'28' : 1, '50' : 2})
        assert(assorter.assort(votes) == 0.5)

def test_cvr_from_dict():
    cvr_dict = [{'id': 1, 'votes': {'AvB': {'Alice':True}, 'CvD': {'Candy':True}}},\
                {'id': 2, 'votes': {'AvB': {'Bob':True}, 'CvD': {'Elvis':True, 'Candy':False}}},\
                {'id': 3, 'votes': {'EvF': {'Bob':1, 'Edie':2}, 'CvD': {'Elvis':False, 'Candy':True}}}]
    cvr_list = CVR.from_dict(cvr_dict)
    assert len(cvr_list) == 3
    assert cvr_list[0].get_id() == 1
    assert cvr_list[1].get_id() == 2
    assert cvr_list[2].get_id() == 3

    assert cvr_list[0].get_votefor('AvB', 'Alice') == True
    assert cvr_list[0].get_votefor('CvD', 'Candy') == True
    assert cvr_list[0].get_votefor('AvB', 'Bob') == False
    assert cvr_list[0].get_votefor('EvF', 'Bob') == False

    assert cvr_list[1].get_votefor('AvB', 'Alice') == False
    assert cvr_list[1].get_votefor('CvD', 'Candy') == False
    assert cvr_list[1].get_votefor('CvD', 'Elvis') == True
    assert cvr_list[1].get_votefor('CvD', 'Candy') == False
    assert cvr_list[1].get_votefor('CvD', 'Edie') == False
    assert cvr_list[1].get_votefor('AvB', 'Bob') == True
    assert cvr_list[1].get_votefor('EvF', 'Bob') == False
                                
    assert cvr_list[2].get_votefor('AvB', 'Alice') == False
    assert cvr_list[2].get_votefor('CvD', 'Candy') == True
    assert cvr_list[2].get_votefor('CvD', 'Edie') == False
    assert cvr_list[2].get_votefor('AvB', 'Bob') == False
    assert cvr_list[2].get_votefor('EvF', 'Bob') == 1
    assert cvr_list[2].get_votefor('EvF', 'Edie') == 2
    assert cvr_list[2].get_votefor('EvF', 'Alice') == False

def test_overstatement():
    mvr_dict = [{'id': 1, 'votes': {'AvB': {'Alice':True}}},\
                {'id': 2, 'votes': {'AvB': {'Bob':True}}},
                {'id': 3, 'votes': {'AvB': {}}},\
                {'id': 4, 'votes': {'CvD': {'Elvis':True, 'Candy':False}}},\
                {'id': 'phantom_1', 'votes': {}, 'phantom': True}]
    mvrs = CVR.from_dict(mvr_dict)

    cvr_dict = [{'id': 1, 'votes': {'AvB': {'Alice':True}}},\
                {'id': 2, 'votes': {'AvB': {'Bob':True}}},\
                {'id': 3, 'votes': {'AvB': {}}},\
                {'id': 4, 'votes': {'CvD': {'Elvis':True}}},\
                {'id': 'phantom_1', 'votes': {}, 'phantom': True}]
    cvrs = CVR.from_dict(cvr_dict)

    winners = ["Alice"]
    losers = ["Bob"]

    aVb = Assertion("AvB", Assorter(contest="AvB", \
                    assort = lambda c, contest="AvB", winr="Alice", losr="Bob":\
                    ( CVR.as_vote(CVR.get_vote_from_cvr("AvB", winr, c)) \
                    - CVR.as_vote(CVR.get_vote_from_cvr("AvB", losr, c)) \
                    + 1)/2, upper_bound = 1))
    assert aVb.overstatement(mvrs[0], cvrs[0], manifest_type="STYLE") == 0
    assert aVb.overstatement(mvrs[0], cvrs[0], manifest_type="ALL") == 0

    assert aVb.overstatement(mvrs[0], cvrs[1], manifest_type="STYLE") == -1
    assert aVb.overstatement(mvrs[0], cvrs[1], manifest_type="ALL") == -1

    assert aVb.overstatement(mvrs[2], cvrs[0], manifest_type="STYLE") == 1/2
    assert aVb.overstatement(mvrs[2], cvrs[0], manifest_type="ALL") == 1/2

    assert aVb.overstatement(mvrs[2], cvrs[1], manifest_type="STYLE") == -1/2
    assert aVb.overstatement(mvrs[2], cvrs[1], manifest_type="ALL") == -1/2


    assert aVb.overstatement(mvrs[1], cvrs[0], manifest_type="STYLE") == 1
    assert aVb.overstatement(mvrs[1], cvrs[0], manifest_type="ALL") == 1

    assert aVb.overstatement(mvrs[2], cvrs[0], manifest_type="STYLE") == 1/2
    assert aVb.overstatement(mvrs[2], cvrs[0], manifest_type="ALL") == 1/2

    assert aVb.overstatement(mvrs[3], cvrs[0], manifest_type="STYLE") == 1
    assert aVb.overstatement(mvrs[3], cvrs[0], manifest_type="ALL") == 1/2

    assert aVb.overstatement(mvrs[3], cvrs[3], manifest_type="STYLE") == 0
    assert aVb.overstatement(mvrs[3], cvrs[3], manifest_type="ALL") == 0
    
    assert aVb.overstatement(mvrs[4], cvrs[4], manifest_type="STYLE") == 1/2
    assert aVb.overstatement(mvrs[4], cvrs[4], manifest_type="ALL") == 1/2
    assert aVb.overstatement(mvrs[4], cvrs[4], manifest_type="ALL") == 1/2
    assert aVb.overstatement(mvrs[4], cvrs[0], manifest_type="STYLE") == 1
    assert aVb.overstatement(mvrs[4], cvrs[0], manifest_type="ALL") == 1
    assert aVb.overstatement(mvrs[4], cvrs[1], manifest_type="STYLE") == 0
    assert aVb.overstatement(mvrs[4], cvrs[1], manifest_type="ALL") == 0
    

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
    assert aVb.overstatement_assorter(mvrs[0], cvrs[0], margin=0.2, manifest_type="STYLE") == 1/1.8
    assert aVb.overstatement_assorter(mvrs[0], cvrs[0], margin=0.2, manifest_type="ALL") == 1/1.8
    
    assert aVb.overstatement_assorter(mvrs[1], cvrs[0], margin=0.2, manifest_type="STYLE") == 0
    assert aVb.overstatement_assorter(mvrs[1], cvrs[0], margin=0.2, manifest_type="ALL") == 0

    assert aVb.overstatement_assorter(mvrs[0], cvrs[1], margin=0.3, manifest_type="STYLE") == 2/1.7
    assert aVb.overstatement_assorter(mvrs[0], cvrs[1], margin=0.3, manifest_type="ALL") == 2/1.7

    assert aVb.overstatement_assorter(mvrs[2], cvrs[0], margin=0.1, manifest_type="STYLE") == 0.5/1.9
    assert aVb.overstatement_assorter(mvrs[2], cvrs[0], margin=0.1, manifest_type="ALL") == 0.5/1.9

def test_cvr_has_contest():
    cvr_dict = [{'id': 1, 'votes': {'AvB': {}, 'CvD': {'Candy':True}}},\
                {'id': 2, 'votes': {'CvD': {'Elvis':True, 'Candy':False}}}]
    cvr_list = CVR.from_dict(cvr_dict)
    assert cvr_list[0].has_contest('AvB')
    assert cvr_list[0].has_contest('CvD')
    assert not cvr_list[0].has_contest('EvF')

    assert not cvr_list[1].has_contest('AvB')
    assert cvr_list[1].has_contest('CvD')
    assert not cvr_list[1].has_contest('EvF')

    
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
    np.testing.assert_almost_equal(TestNonnegMean.kaplan_wald(s, g=0.1, random_order = False),\
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

def test_initial_sample_size():
    N_cards = int(10**3)
    risk_function = lambda x: TestNonnegMean.kaplan_kolmogorov(x, N=N_cards, t=1/2, g=0.1) 
    n_det = TestNonnegMean.initial_sample_size(risk_function, N_cards, 0.1, 0.001)
    n_rand = TestNonnegMean.initial_sample_size(risk_function, N_cards, 0.1, 0.001, reps=100)
    print(n_det, n_rand)

    # This tests whether, in a simple example in which null hypothesis is true, 
    # the distribution of p-values is dominated by the uniform distribution, 
    # that is, Prob(p \le x) \le x, x \in [0, 1].
    # VT: I have just added some basic sanity checks to ensure that p is small
    # when the claimed mean is much higher than the data suggests.
    # TODO add some tests that use the specific values we expect.
    
def test_initial_sample_size_KW():
    # Test the initial_sample_size on the Kaplan-Wald risk function
    g = 0
    alpha = 0.05
    error_rate = 0.01
    N = int(10**4)
    margin = 0.1
    one_over = 1/3.8 # 0.5/(2-margin)
    clean = 1/1.9    # 1/(2-margin)

    risk_function = lambda x: TestNonnegMean.kaplan_wald(x, t=1/2, g=g, random_order=False)
    # first test
    bias_up = False
    sam_size = TestNonnegMean.initial_sample_size(risk_function, N, margin, error_rate, alpha=alpha, t=1/2, reps=None,\
                            bias_up=bias_up, quantile=0.5, seed=1234567890)
    sam_size_0 = 59 # math.ceil(math.log(20)/math.log(2/1.9)), since this is < 1/error_rate
    np.testing.assert_almost_equal(sam_size, sam_size_0)
    # second test
    bias_up = True
    sam_size = TestNonnegMean.initial_sample_size(risk_function, N, margin, error_rate, alpha=alpha, t=1/2, reps=None,\
                            bias_up=bias_up, quantile=0.5, seed=1234567890)
    sam_size_1 = 72 # (1/1.9)*(2/1.9)**71 = 20.08
    np.testing.assert_almost_equal(sam_size, sam_size_1)
    # third test
    sam_size = TestNonnegMean.initial_sample_size(risk_function, N, margin, error_rate, alpha=alpha, t=1/2, reps=1000,\
                            bias_up=bias_up, quantile=0.5, seed=1234567890)
    np.testing.assert_array_less(sam_size_0, sam_size+1) # crude test, but ballpark
    np.testing.assert_array_less(sam_size, sam_size_1+1) # crude test, but ballpark
    
def test_kaplan_martingale():
    eps = 0.0001  # Generic small value for use when not sure exactly how small it should be.
    
    # When all the items are ones, estimated p for a mean of 1 should be 1.
    s = np.ones(5)
    np.testing.assert_almost_equal(TestNonnegMean.kaplan_martingale(s, N=100000, t=1, random_order = True)[0],1.0)

    # When t is much smaller than all the sample data, p should be very small.
    # VT: Note I am not sure exactly how small.
    np.testing.assert_array_less(TestNonnegMean.kaplan_martingale(s, N=100000, t=0, random_order = True)[:1],[eps])

    s = [0.6,0.8,1.0,1.2,1.4]
    np.testing.assert_array_less(TestNonnegMean.kaplan_martingale(s, N=100000, t=0, random_order = True)[:1],[eps])

    s1 = [1, 0, 1, 1, 0, 0, 1]
    kmart1 = TestNonnegMean.kaplan_martingale(s1,
                                              N=7,
                                              t=3/7,
                                              random_order=True)[1]
    # No nans introduced
    assert(not any(np.isnan(kmart1)))

    s2 = [1, 0, 1, 1, 0, 0, 0]
    kmart2 = TestNonnegMean.kaplan_martingale(s2,
                                              N=7,
                                              t=3/7,
                                              random_order=True)[1]
    # Since s1 and s2 only differ in the last observation,
    # the resulting martingales should be identical up to the second
    # last entry.
    assert(all(np.equal(kmart2[0:(len(kmart2)-1)],
                        kmart1[0:(len(kmart1)-1)])))

def test_assorter_mean():
    pass # [FIX ME]

def test_cvr_from_raire():
    raire_cvrs = [['1'],\
                  ["Contest","339","5","15","16","17","18","45"],\
                  ["339","99813_1_1","17"],\
                  ["339","99813_1_3","16"],\
                  ["339","99813_1_6","18","17","15","16"],\
                  ["3","99813_1_6","2"]\
                 ]
    c = CVR.from_raire(raire_cvrs)
    assert len(c) == 3
    assert c[0].id == "99813_1_1"
    assert c[0].votes == {'339': {'17':1}}
    assert c[2].id == "99813_1_6"
    assert c[2].votes == {'339': {'18':1, '17':2, '15':3, '16':4}, '3': {'2':1}} # merges votes?

if __name__ == "__main__":
    test_cvr_from_raire()
    test_cvr_from_dict()
    test_cvr_has_contest()

    test_make_plurality_assertions()
    test_supermajority_assorter()

    test_overstatement()
    test_overstatement_assorter()
    
    test_rcv_lfunc_wo()
    test_rcv_votefor_cand()    
    test_rcv_assorter()
    
    test_kaplan_markov()
    test_kaplan_wald()
    test_kaplan_kolmogorov()
    test_initial_sample_size()
    test_initial_sample_size_KW()
