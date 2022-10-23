import math
import numpy as np
import json
import csv
import warnings
import typing
from numpy import testing
from collections import OrderedDict, defaultdict
from cryptorandom.cryptorandom import SHA256, random, int_from_hash
from cryptorandom.sample import random_permutation
from cryptorandom.sample import sample_by_index

##########################################################################################    
class CVR:
    '''
    Generic class for cast-vote records.

    The CVR class DOES NOT IMPOSE VOTING RULES. For instance, the social choice
    function might consider a CVR that contains two votes in a contest to be an overvote.

    Rather, a CVR is supposed to reflect what the ballot shows, even if the ballot does not
    contain a valid vote in one or more contests.

    Class method get_vote_for returns the vote for a given candidate if the candidate is a
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

    CVRs can be flagged as "phantoms" to account for cards not listed in the manifest (Boolean `phantom` attribute).

    CVRs can include sampling probabilities `p` and sample numbers `sample_num` (random numbers to facilitate
        consistent sampling)

    CVRs can include a sequence number to facilitate ordering and reordering

    Methods:
    --------

    get_vote_for: 
         get_vote_for(candidate, contest_id) returns the value in the votes dict for the key `candidate`, or
         False if the candidate did not get a vote or the contest_id is not in the CVR
    has_contest: returns bool
         does the CVR have the contest?
    cvrs_to_json:
         represent CVR list as json
    from_dict: create a CVR from a dict
    from_dict_of_dicts:
         create dict of CVRs from a list of dicts
    from_raire:
         create CVRs from the RAIRE representation
    '''

    def __init__(self, id = None, votes = {}, phantom=False, sample_num=None, p=None, sampled=False):
        self.votes = votes
        self.id = id
        self.phantom = phantom
        self.sample_num = sample_num
        self.p = p
        self.sampled = sampled

    def __str__(self):
        return f"id: {str(self.id)} votes: {str(self.votes)} phantom: {str(self.phantom)}"

    def get_vote_for(self, contest_id, candidate):
        return (False if (contest_id not in self.votes or candidate not in self.votes[contest_id])
                else self.votes[contest_id][candidate])

    def has_contest(self, contest_id: str):
        return contest_id in self.votes

    def has_one_vote(self, contest_id, candidates):
        '''
        Is there exactly one vote among the candidates in the contest with contest_id?

        Parameters:
        -----------
        contest_id: string
            identifier of contest
        candidates: list
            list of identifiers of candidates

        Returns:
        ----------
        True if there is exactly one vote among those candidates in that contest, where a
        vote means that the value for that key casts as boolean True.
        '''
        v = np.sum([0 if c not in self.votes[contest_id] else bool(self.votes[contest_id][c]) \
                    for c in candidates])
        return True if v==1 else False

    def rcv_lfunc_wo(self, contest_id: str, winner: str, loser: str):
        '''
        Check whether vote is a vote for the loser with respect to a 'winner only'
        assertion between the given 'winner' and 'loser'.

        Parameters:
        -----------
        contest_id: string
            identifier for the contest
        winner: string
            identifier for winning candidate
        loser: string
            identifier for losing candidate
        cvr: CVR object

        Returns:
        --------
        1 if the given vote is a vote for 'loser' and 0 otherwise
        '''
        rank_winner = self.get_vote_for(contest_id, winner)
        rank_loser = self.get_vote_for(contest_id, loser)

        if not bool(rank_winner) and bool(rank_loser):
            return 1
        elif bool(rank_winner) and bool(rank_loser) and rank_loser < rank_winner:
            return 1
        else:
            return 0

    def rcv_votefor_cand(self, contest_id: str, cand: str, remaining: list):
        '''
        Check whether 'vote' is a vote for the given candidate in the context
        where only candidates in 'remaining' remain standing.

        Parameters:
        -----------
        contest_id: string
            identifier of the contest used in the CVRs
        cand: string
            identifier for candidate
        remaining: list
            list of identifiers of candidates still standing

        vote: dict of dicts

        Returns:
        --------
        1 if the given vote for the contest counts as a vote for 'cand' and 0 otherwise. Essentially,
        if you reduce the ballot down to only those candidates in 'remaining',
        and 'cand' is the first preference, return 1; otherwise return 0.
        '''
        if not cand in remaining:
            return 0

        if not bool(rank_cand:= self.get_vote_for(contest_id, cand)):
            return 0
        else:
            for altc in remaining:
                if altc == cand:
                    continue
                rank_altc = self.get_vote_for(contest_id, altc)
                if bool(rank_altc) and rank_altc <= rank_cand:
                    return 0
            return 1

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
        raire: list of comma-separated values
            source in RAIRE format. From the RAIRE documentation:
            The RAIRE format (for later processing) is a CSV file.
            First line: number of contests.
            Next, a line for each contest
             Contest,id,N,C1,C2,C3 ...
                id is the contest_id
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
            contest_id = c[0]
            id = c[1]
            votes = {}
            for j in range(2, len(c)):
                votes[str(c[j])] = j-1
            cvr_list.append(CVR.from_vote(votes, id=id, contest_id=contest_id, phantom=phantom))
        return CVR.merge_cvrs(cvr_list)

    @classmethod
    def from_raire_file(cls, cvr_file: str=None):
        '''
        Read CVR data from a file; construct list of CVR objects from the data
        
        Parameters
        ----------
        cvr_file : str
            filename 
            
        Returns
        -------
        cvrs: list of CVR objects
        cvrs_read: int
            number of CVRs read
        unique_ids: int
            number of distinct CVR identifiers read
        '''
        cvr_in = []
        with open(cvr_file) as f:
            cvr_reader = csv.reader(f, delimiter=',', quotechar='"')
            for row in cvr_reader:
                cvr_in.append(row)
        cvrs = CVR.from_raire(cvr_in)
        return cvrs, len(cvr_in), len(cvrs)
        
    @classmethod
    def merge_cvrs(cls, cvr_list):
        '''
        Takes a list of CVRs that might contain duplicated ballot ids and merges the votes
        so that each identifier is listed only once, and votes from different records for that
        identifier are merged.
        The merge is in the order of the list: if a later mention of a ballot id has votes
        for the same contest as a previous mention, the votes in that contest are updated
        per the later mention.

        If any of the CVRs has phantom==False, sets phantom=False in the result.


        Parameters:
        -----------
        cvr_list: list of CVRs

        Returns:
        -----------
        list of merged CVRs
        '''
        od = OrderedDict()
        for c in cvr_list:
            if c.id not in od:
                od[c.id] = c
            else:
                od[c.id].votes = {**od[c.id].votes, **c.votes}
                od[c.id].phantom = (c.phantom and od[c.id].phantom)
        return [v for v in od.values()]

    @classmethod
    def from_vote(cls, vote, id: object=1, contest_id: str='AvB', phantom: bool=False):
        '''
        Wraps a vote and creates a CVR, for unit tests

        Parameters:
        ----------
        vote: dict of votes in one contest
        id: str
            CVR id
        contest_id: str
            identifier of the contest

        Returns:
        --------
        CVR containing that vote in the contest "AvB", with CVR id=1.
        '''
        return CVR(id=id, votes={contest_id: vote}, phantom=phantom)

    @classmethod
    def as_vote(cls, v):
        return int(bool(v))

    @classmethod
    def as_rank(cls, v):
        return int(v)


    @classmethod
    def make_phantoms(cls, audit: None, contests: dict=None, cvr_list: list=None, prefix: str='phantom-'):
        '''
        Make phantom CVRs as needed for phantom cards; set contest parameters `cards` (if not set) and `cvrs`

        **Currently only works for unstratified audits.**
        If `audit.strata[s]['use_style']`, phantoms are "per contest": each contest needs enough to account for the 
        difference between the number of cards that might contain the contest and the number of CVRs that contain 
        the contest. This can result in having more cards in all (manifest and phantoms) than max_cards, the maximum cast.

        If `not use_style`, phantoms are for the election as a whole: need enough to account for the difference
        between the number of cards in the manifest and the number of CVRs that contain the contest. Then, the total
        number of cards (manifest plus phantoms) equals max_cards.

        If `not use_style` sets `cards = max_cards` for each contest

        Parameters
        ----------
        cvr_list: list of CVR objects
            the reported CVRs
        contests: dict of contests
            information about each contest under audit
        prefix: String
            prefix for ids for phantom CVRs to be added

        Returns
        -------
        cvr_list: list of CVR objects
            the reported CVRs and the phantom CVRs
        n_phantoms: int
            number of phantom cards added

        Side effects
        ------------
        for each contest in `contests`, sets `cards` to max_cards if not specified by the user or if `not use_style`
        for each contest in `contests`, set `cvrs` to be the number of (real) CVRs that contain the contest
        '''
        if len(audit.strata) > 1:
            raise NotImplementedError('stratified audits not implemented')
        stratum = next(iter(audit.strata.values()))
        use_style = stratum.use_style
        max_cards = stratum.max_cards
        phantom_vrs = []
        n_cvrs = len(cvr_list)
        for c, v in contests.items():  # set contest parameters
            v.cvrs = np.sum([cvr.has_contest(v.id) for cvr in cvr_list if not cvr.phantom])
            v.cards = max_cards if ((v.cards is None) or (not use_style)) else v.cards
        # Note: this will need to change for stratified audits
        if not use_style:              #  make (max_cards - len(cvr_list)) phantoms
            phantoms = max_cards - n_cvrs
            for i in range(phantoms):
                phantom_vrs.append(CVR(id=prefix+str(i+1), votes={}, phantom=True))
        else:                          # create phantom CVRs as needed for each contest
            for c, v in contests.items():
                phantoms_needed = v.cards - v.cvrs
                while len(phantom_vrs) < phantoms_needed:  # creat additional phantoms
                    phantom_vrs.append(CVR(id=prefix+str(len(phantom_vrs)+1), votes={}, phantom=True))
                for i in range(phantoms_needed):
                    phantom_vrs[i].votes[v.id]={}  # list contest c on the phantom CVR
            phantoms = len(phantom_vrs)
        cvr_list = cvr_list + phantom_vrs
        return cvr_list, phantoms

    @classmethod
    def assign_sample_nums(cls, cvr_list, prng):
        '''
        Assigns a pseudo-random sample number to each cvr in cvr_list

        Parameters
        ----------
        cvr_list: list of CVR objects
        prng: instance of cryptorandom SHA256 generator

        Returns
        -------
        True

        Side effects
        ------------
        assigns (or overwrites) sample numbers in each CVR in cvr_list
        '''
        for cvr in cvr_list:
            cvr.sample_num = int_from_hash(prng.nextRandom())
        return True


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
        True
        
        Side effects
        ------------
        cvr_list is sorted by sample_num
        '''
        cvr_list.sort(key = lambda x: x.sample_num)
        return True

    @classmethod
    def consistent_sampling(cls, cvr_list: list=None, contests: dict=None, sampled_cvr_indices: list=None) -> list:
        '''
        Sample CVR ids for contests to attain sample sizes in sample_size_dict

        Assumes that phantoms have already been generated and sample_num has been assigned
        to every CVR, including phantoms

        Parameters
        ----------
        cvr_list: list
            list of CVR objects
        contests: dict
            dict of Contest objects. Contest sample sizes must be set before calling this function.
        sampled_cvr_indices: list
            indices of cvrs already in the sample

        Returns
        -------
        sampled_cvr_indices: list
            indices of CVRs to sample (0-indexed)
        '''
        current_sizes = defaultdict(int)
        contest_in_progress = lambda c: (current_sizes[c.id] < c.sample_size)
        if sampled_cvr_indices is None:
            sampled_cvr_indices = []
        else:
            for sam in sampled_cvr_indices:
                for c, v in contests.items():
                    current_sizes[c] += (1 if cvr_list[sam].has_contest(v.id) else 0)
        sorted_cvr_indices = [i for i, cv in sorted(enumerate(cvr_list), key = lambda x: x[1].sample_num)]
        inx = len(sampled_cvr_indices)
        while any([contest_in_progress(v) for c, v in contests.items()]):
            if any([(contest_in_progress(v) and cvr_list[sorted_cvr_indices[inx]].has_contest(v.id)) 
                     for c, v in contests.items()]):
                sampled_cvr_indices.append(sorted_cvr_indices[inx])
                for c, v in contests.items():
                    current_sizes[c] += (1 if cvr_list[sorted_cvr_indices[inx]].has_contest(v.id) else 0)
            inx += 1
        for i in range(len(cvr_list)):
            if i in sampled_cvr_indices:
                cvr_list[i].sampled = True
        return sampled_cvr_indices

    @classmethod
    def tabulate_styles(cls, cvr_list: list=None):
        '''
        tabulate unique CVR styles in cvr_list

        Parameters
        ----------
        cvr_list: a list of CVR objects

        Returns
        -------
        a dict of styles and the counts of those styles 
        '''
        # iterate through and find all the unique styles
        style_counts = defaultdict(int)
        for cvr in cvr_list:
            style_counts[set(cvr.votes.keys())] += 1
        return style_counts

    @classmethod
    def count_votes(cls, cvr_list: list=None):
        """
        tabulate total votes for each candidate in each contest in cvr_list.
        For plurality, supermajority, and approval. Not useful for ranked-choice voting

        Parameters
        ----------
        cvr_list: list of CVR objects 

        Returns
        -------
        dict of dicts:
            main key is contest
            sub key is the candidate in the contest
            value is the number of votes for that candidate in that contest
        """
        d = defaultdict(lambda: defaultdict(int))
        for c in cvr_list:
            for con, votes in c.votes.items():
                for cand in votes:
                    d[con][cand] += CVR.as_vote(c.get_vote_for(con, cand))
        return d
