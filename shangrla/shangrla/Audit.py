import math
import numpy as np
import json
import csv
import warnings
from collections import OrderedDict, defaultdict
from cryptorandom.cryptorandom import SHA256, random, int_from_hash
from cryptorandom.sample import random_permutation
from cryptorandom.sample import sample_by_index
from .NonnegMean import NonnegMean

##########################################################################################
class Stratum:
    '''
    stratum attributes
    '''

    def __init__(
                  self,
                  id: str=None,
                  max_cards: int=None,
                  use_style: bool=None,
                  replacement: bool=False,
                  audit_type:  str=None,
                  test:  callable=None,
                  estimator:  callable=None,
                  bet: callable=None,
                  test_kwargs: dict=None):
        self.id = id
        self.max_cards = max_cards
        self.use_style = use_style
        self.replacement = replacement
        self.audit_type = audit_type
        self.test = test
        self.estimator = estimator
        self.bet = bet
        self.test_kwargs = test_kwargs

    @classmethod
    def from_dict(cls, d: dict=None):
        s = Stratum()
        s.__dict__.update(d)
        return s

##########################################################################################
class NpEncoder(json.JSONEncoder):
    '''
    for json dumps of Audit, Assertion, Contest
    '''
    def default(self, obj):
        if isinstance(obj, np.integer):
            return int(obj)
        if isinstance(obj, np.floating):
            return float(obj)
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        if isinstance(obj, Assertion):
            return obj.__str__()
        if isinstance(obj, Audit):
            return obj.__str__()
        if isinstance(obj, Contest):
            return obj.__str__()
        return super(NpEncoder, self).default(obj)

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

    CVRs can be flagged as "phantoms" to account for cards not listed in the manifest (Boolean 
    `phantom` attribute).

    CVRs can include sampling probabilities `p` and sample numbers `sample_num` (pseudo-random numbers 
    to facilitate consistent sampling)

    CVRs can include a sequence number to facilitate ordering, sorting, and permuting

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
        for c, con in contests.items():  # set contest parameters
            con.cvrs = np.sum([cvr.has_contest(con.id) for cvr in cvr_list if not cvr.phantom])
            con.cards = max_cards if ((con.cards is None) or (not use_style)) else con.cards
        # Note: this will need to change for stratified audits
        if not use_style:              #  make (max_cards - len(cvr_list)) phantoms
            phantoms = max_cards - n_cvrs
            for i in range(phantoms):
                phantom_vrs.append(CVR(id=prefix+str(i+1), votes={}, phantom=True))
        else:                          # create phantom CVRs as needed for each contest
            for c, con in contests.items():
                phantoms_needed = con.cards - con.cvrs
                while len(phantom_vrs) < phantoms_needed:  # creat additional phantoms
                    phantom_vrs.append(CVR(id=prefix+str(len(phantom_vrs)+1), votes={}, phantom=True))
                for i in range(phantoms_needed):
                    phantom_vrs[i].votes[con.id]={}  # list contest c on the phantom CVR
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
        prepare the MVRs and CVRs for comparison by putting them into the same (random) order
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
        Sample CVR ids for contests to attain sample sizes in contests, a dict of Contest objects

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
                for c, con in contests.items():
                    current_sizes[c] += (1 if cvr_list[sam].has_contest(con.id) else 0)
        sorted_cvr_indices = [i for i, cv in sorted(enumerate(cvr_list), key = lambda x: x[1].sample_num)]
        inx = len(sampled_cvr_indices)
        while any([contest_in_progress(con) for c, con in contests.items()]):
            if any([(contest_in_progress(con) and cvr_list[sorted_cvr_indices[inx]].has_contest(con.id))
                     for c, con in contests.items()]):
                sampled_cvr_indices.append(sorted_cvr_indices[inx])
                for c, con in contests.items():
                    if cvr_list[sorted_cvr_indices[inx]].has_contest(con.id) and contest_in_progress(con):
                        con.sample_threshold = cvr_list[sorted_cvr_indices[inx]].sample_num
                        current_sizes[c] += 1
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
            style_counts[frozenset(cvr.votes.keys())] += 1
        return style_counts

    @classmethod
    def tabulate_votes(cls, cvr_list: list=None):
        """
        tabulate total votes for each candidate in each contest in cvr_list.
        For plurality, supermajority, and approval. Not useful for ranked-choice voting.

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

    @classmethod
    def tabulate_cards_contests(cls, cvr_list: list=None):
        """
        Tabulate the number of cards containing each contest

        Parameters
        ----------
        cvr_list: list of CVR objects

        Returns
        -------
        dict:
            main key is contest
            value is the number of cards containing that contest
        """
        d = defaultdict(int)
        for c in cvr_list:
            for con in c.votes:
                d[con] += 1
        return d

##########################################################################################
class Audit:
    '''
    Various constants that specify what kind of contests are audited and how to audit them.
    Methods to estimate the sample size to audit every contest.
    Methods for logging and checking parameters.
    '''

    ATTRIBUTES = ('seed', 'sim_seed', 'cvr_file', 'manifest_file', 'sample_file', 'mvr_file',
                  'log_file', 'quantile', 'error_rate_1', 'error_rate_2', 'reps', 'max_cards',
                  'strata')


    class AUDIT_TYPE:
        '''
        types of audit
        '''
        AUDIT_TYPES = (POLLING:= 'POLLING',
                       CARD_COMPARISON:= 'CARD_COMPARISON'
                      )
        # TO DO: BATCH_COMPARISON, STRATIFIED, HYBRID, ...


    def __init__(
                 self,
                 seed: object=None,
                 sim_seed: int=123456789,
                 cvr_file: str=None,
                 manifest_file: str=None,
                 sample_file: str=None,
                 mvr_file: str=None,
                 log_file: str=None,
                 quantile: float=0.9,
                 error_rate_1: float=0.001,
                 error_rate_2: float=0,
                 reps: int=None,
                 max_cards: int=None,
                 strata: dict=None):
        self.seed = seed
        self.sim_seed = sim_seed
        self.cvr_file = cvr_file,
        self.manifest_file = manifest_file
        self.sample_file = sample_file
        self.mvr_file = mvr_file
        self.log_file = log_file
        self.quantile = quantile
        self.error_rate_1 = error_rate_1
        self.error_rate_2 = error_rate_2
        self.reps = reps
        self.max_cards = max_cards
        self.strata = strata

    def __str__(self):
        return str(self.__dict__)

    def find_sample_size(self, contests: dict=None, cvrs: list=None, mvr_sample: list=None, cvr_sample: list=None) -> int:
        '''
        Estimate sample size for each contest and overall to allow the audit to complete.
        Uses simulations. For speed, uses the numpy.random Mersenne Twister instead of cryptorandom.

        Parameters
        ----------
        contests: dict of dicts
            the contest data structure. outer keys are contest identifiers; inner keys are assertions
        cvrs: list of CVR objects
            the full set of CVRs
        mvr_sample: list of CVR objects
            manually ascertained votes
        cvr_sample: list of CVR objects
            CVRs corresponding to the cards that were manually inspected

        Returns
        -------
        new_size: int
            new sample size

        Side effects
        ------------
        sets c.sample_size for each Contest in contests
        if use_style, sets cvr.p for each CVR
        '''
        # Currently, only unstratified audits are supported
        if len(self.strata) > 1:
            raise NotImplementedError('Stratified audits are not currently implemented.')
        stratum = next(iter(self.strata.values())) # the only stratum
        if stratum.use_style and cvrs is None:
            raise ValueError("stratum.use_style==True but cvrs were not provided.")
        # unless style information is being used, the sample size is the same for every contest.
        old = (0 if stratum.use_style 
               else len(mvr_sample))
        old_sizes = {c:old for c in contests.keys()}
        for c, con in contests.items():
            if stratum.use_style:
                old_sizes[c] = np.sum(np.array([cvr.sampled for cvr in cvrs if cvr.has_contest(c)]))
            new_size = 0
            for a, asn in con.assertions.items():
                if not asn.proved:
                    if mvr_sample is not None: # use MVRs to estimate the next sample size. Set `prefix=True` to use data
                        data, u =  asn.mvrs_to_data(mvr_sample, cvr_sample)
                        new_size = max(new_size, asn.find_sample_size(data=data, prefix=True,
                                                                  reps=self.reps, quantile=self.quantile,
                                                                  seed=self.sim_seed))
                    else:
                        data=None
                        new_size = max(new_size, asn.find_sample_size(data=data, rate_1=self.error_rate_1,
                                                                  rate_2=self.error_rate_2,
                                                                  reps=self.reps, quantile=self.quantile,
                                                                  seed=self.sim_seed))
            con.sample_size = new_size
        if stratum.use_style:
            for cvr in cvrs:
                if cvr.sampled:
                    cvr.p=1
                else:
                    cvr.p=0
                    for c, con in contests.items():
                        if cvr.has_contest(c) and not cvr.sampled:
                            cvr.p = max(con.sample_size/(con.cards - old_sizes[c]), cvr.p)
            total_size = math.ceil(np.sum([x.p for x in cvrs if not x.phantom]))
        else:
            total_size = np.max(np.array([con.sample_size for con in contests.values()]))
        return total_size

    def check_audit_parameters(self, contests: dict=None):
        '''
        Check whether the audit parameters are valid; complain if not.

        Parameters
        ----------
        contests: dict of Contests
            contest-specific information for the audit

        Returns
        -------

        Side effects
        ------------
        raises exceptions if Audit parameters or contest parameters fail tests
        '''
        assert self.error_rate_1 >= 0, 'expected rate of 1-vote errors must be nonnegative'
        assert self.error_rate_2 >= 0, 'expected rate of 2-vote errors must be nonnegative'
        for c, con in contests.items():
            assert con.risk_limit > 0, f'risk limit {con.risk_limit} negative in contest {c}'
            assert con.risk_limit <= 1/2, f'risk limit {con.risk_limit} exceeds 1/2 in contest {c}'
            assert con.choice_function in Contest.SOCIAL_CHOICE_FUNCTION.SOCIAL_CHOICE_FUNCTIONS, \
                      f'unsupported choice function {con.choice_function} in contest {c}'
            assert con.n_winners <= len(con.candidates), f'more winners than candidates in contest {c}'
            assert len(con.winner) == con.n_winners, \
                f'number of reported winners does not equal n_winners in contest {c}'
            for w in con.winner:
                assert w in con.candidates, f'reported winner {w} is not a candidate in contest {c}'
            if con.choice_function in [Contest.SOCIAL_CHOICE_FUNCTION.IRV,
                                     Contest.SOCIAL_CHOICE_FUNCTION.SUPERMAJORITY]:
                assert con.n_winners == 1, f'{con.choice_function} can have only 1 winner in contest {c}'
            if con.choice_function == Contest.SOCIAL_CHOICE_FUNCTION.IRV:
                assert con.assertion_file, f'IRV contest {c} requires an assertion file'


    def write_audit_parameters(self, contests: dict=None):
        '''
        Write audit parameters as a json structure

        Parameters:
        ---------
        audit: Audit
            general information about the audit

        contests: dict of dicts
            contest-specific information for the audit

        Returns:
        --------
        no return value
        '''
        log_file = self.log_file
        out = {'Audit': self,
               'contests': contests
              }
        with open(log_file, 'w') as f:
            f.write(json.dumps(out, cls=NpEncoder))

    def summarize_status(self, contests: dict=None):
        '''
        Determine whether the audit of individual assertions, contests, and the whole election are finished.

        Prints a summary.

        Parameters:
        -----------
        audit: Audit
            general information about the audit
        contests: dict of Contest objects
            dict of contest information

        Returns:
        --------
        done: boolean
            is the audit finished?
        '''
        done = True
        for c, con in contests.items():
            print(f'\np-values for assertions in contest {c}')
            cpmax = 0
            for i, a in con.assertions.items():
                cpmax = np.max([cpmax, a.p_value])
                print(f'\t{i}: {a.p_value}')
            if cpmax <= contests[c].risk_limit:
                print(f'\ncontest {c} AUDIT COMPLETE at risk limit {con.risk_limit}. Attained risk {cpmax}')
            else:
                done = False
                print(f'\ncontest {c} audit INCOMPLETE at risk limit {con.risk_limit}. Attained risk {cpmax}')
                print("assertions remaining to be proved:")
                for i, a in con.assertions.items():
                    if a.p_value > con.risk_limit:
                        print(f'\t{i}\t{a}: current risk {a.p_value}')
        return done

    @classmethod
    def from_dict(cls, d: dict=None):
        '''
        make an Audit object from a dict of attributes
        assumes that the 'strata' attribute is itself a dict
        '''
        a = Audit()
        # unpack stratum dictionary and create Stratum objects
        strat_obj = {}
        for id, st in d['strata'].items():
            strat_obj[id] = Stratum.from_dict(st)
            strat_obj[id].id = id
        d['strata'] = strat_obj
        a.__dict__.update(d)
        return a


##########################################################################################
class Assertion:
    '''
    Objects and methods for SHANGRLA assertions about election outcomes

    An _assertion_ is a statement of the form
      "the average value of this assorter applied to the ballots is greater than 1/2"
    An _assorter_ maps votes to nonnegative numbers not exceeding some upper bound, `upper_bound`
    '''
    # supported json assertion types for imported assertions
    JSON_ASSERTION_TYPES = (WINNER_ONLY:= "WINNER_ONLY",
                            IRV_ELIMINATION:= "IRV_ELIMINATION")

    def __init__(
                 self,
                 contest: object=None,
                 assorter: callable=None,
                 winner: str=None,
                 loser: str=None,
                 margin: float=None,
                 test: object=None,
                 p_value: float=1,
                 p_history: list=[],
                 proved: bool=False,
                 sample_size: int=None):
        '''
        test is an instance of NonnegMean

        Parameters
        ----------
        contest: Contest instance
            contest to which the assorter is relevant
        winner: str
            identifier for the nominal "winner" for this assertion. Can be an element of self.contest.candidates,
            an element of Contest.CANDIDATES, or an arbitrary label.
            Using an element of self.contest.candidates or an element of Contest.CANDIDATES can be useful for
            setting the margin in approval, plurality, and supermajority contests.
        loser: str
            identifier for the nominal "loser" for this assertion. Can be an element of self.contest.candidates,
            an element of Contest.CANDIDATES, or an arbitrary label.
            Using an element of self.contest.candidates or an element of Contest.CANDIDATES can be useful for
            setting the margin in approval, plurality, and supermajority contests.
        assorter: Assorter instance
            the assorter for the assertion
        margin: float
            the assorter margin. Generally this will not be known when the assertion is created, but will be set
            later.
        test: instance of class NonnegMean
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
        self.contest = contest
        self.winner = winner
        self.loser = loser
        self.assorter = assorter
        self.margin = margin
        self.test = test
        self.p_value = p_value
        self.p_history = p_history
        self.proved = proved
        self.sample_size = sample_size

    def __str__(self):
        return (f'contest_id: {self.contest.id} winner: {self.winner} loser: {self.loser} '
                f'assorter: {str(self.assorter)} p-value: {self.p_value} '
                f'margin: {self.margin} test: {str(self.test)} '
                f'p-history length: {len(self.p_history)} proved: {self.proved} sample_size: {self.sample_size} '
                f'assorter upper bound: {self.assorter.upper_bound}'
               )

    def min_p(self):
        return min(self.p_history)


    def margin(self, cvr_list: list=None, use_style: bool=True):
        '''
        find the margin for a list of Cvrs.
        By definition, the margin is twice the mean of the assorter, minus 1.

        Parameters
        ----------
        cvr_list: list
            a list of cast-vote records

        Returns
        ----------
        margin: float
        '''
        return 2*self.assorter.mean(cvr_list, use_style=use_style)-1

    def overstatement_assorter_margin(self, error_rate_1: float=0, error_rate_2: float=0) -> float:
        '''
        find the overstatement assorter margin corresponding to an assumed rate of 1-vote and 2-vote overstatements

        Parameters
        ----------
        error_rate_1: float
            the assumed rate of one-vote overstatement errors in the CVRs
        error_rate_2: float
            the assumed rate of two-vote overstatement errors in the CVRs

        Returns
        -------
        the overstatement assorter margin implied by the reported margin and the assumed rates of overstatements
        '''
        return (1 - (error_rate_2 + error_rate_1/2)*self.assorter.upper_bound/self.margin) \
                /(2*self.assorter.upper_bound/self.margin-1)

    def overstatement_assorter_mean(self, error_rate_1: float=0, error_rate_2: float=0) -> float:
        '''
        find the overstatement assorter mean corresponding to assumed rates of 1-vote and 2-vote overstatements

        Parameters
        ----------
        error_rate_1: float
            the assumed rate of one-vote overstatement errors in the CVRs
        error_rate_2: float
            the assumed rate of two-vote overstatement errors in the CVRs


        Returns
        -------
        overstatement assorter mean implied by the assorter mean and the assumed error rates
        '''
        return (1-error_rate_1/2 - error_rate_2)/(2-self.margin/self.assorter.upper_bound)


    def overstatement_assorter(self, mvr: list=None, cvr: list=None, use_style=True) -> float:
        '''
        assorter that corresponds to normalized overstatement error for an assertion

        If `use_style == True`, then if the CVR contains the contest but the MVR does not,
        that is considered to be an overstatement, because the ballot is presumed to contain
        the contest.

        If `use_style == False`, then if the CVR contains the contest but the MVR does not,
        the MVR is considered to be a non-vote in the contest.

        Parameters
        -----------
        mvr: Cvr
            the manual interpretation of voter intent
        cvr: Cvr
            the machine-reported cast vote record.

        Returns
        --------
        over: float
            (1-o/u)/(2-v/u), where
                o is the overstatement
                u is the upper bound on the value the assorter assigns to any ballot
                v is the assorter margin
        '''
        return (1-self.assorter.overstatement(mvr, cvr, use_style)
                / self.assorter.upper_bound)/(2-self.margin/self.assorter.upper_bound)

    def set_margin_from_cvrs(self, audit: object=None, cvr_list: list=None):
        '''
        find assorter margin from cvrs and store it

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
        if len(audit.strata) > 1:
            raise NotImplementedError('stratified audits not yet supported')
        stratum = next(iter(audit.strata.values()))
        use_style = stratum.use_style
        amean =self.assorter.mean(cvr_list, use_style=use_style)
        if amean < 1/2:
            warnings.warn(f"assertion {self} not satisfied by CVRs: mean value is {amean}")
        self.margin = 2*amean-1
        if self.contest.audit_type == Audit.AUDIT_TYPE.POLLING:
            self.test.u = self.assorter.upper_bound
        elif self.contest.audit_type == Audit.AUDIT_TYPE.CARD_COMPARISON:
            self.test.u = 2/(2-self.margin/self.assorter.upper_bound)
        else:
            raise NotImplementedError(f'audit type {self.contest.audit_type} not supported')

    def find_margin_from_tally(self, tally: dict=None):
        '''
        find the assorter margin between implied by a tally.

        Generally useful only for approval, plurality, and supermajority contests.

        Assumes the number of cards containing the contest has been set.

        Parameters
        ----------
        tally: dict
            dict of tallies for the candidates in the contest. Keys are candidates as listed
            in Contest.candidates. If `tally is None` tries to use the contest.tally.

        The margin for a supermajority contest with a winner is (see SHANRGLA section 2.3)
              2(pq/(2f) + (1 − q)/2 - 1/2) = q(p/f-1), where:
                     q is the fraction of cards that have valid votes
                     p is the fraction of cards that have votes for the winner
                     f is the fraction of valid votes required to win.

        Returns
        -------
        nothing

        Side effects
        ------------
        sets self.margin

        '''
        tally = tally if tally else self.contest.tally
        if self.contest.choice_function == Contest.SOCIAL_CHOICE_FUNCTION.PLURALITY \
             or self.contest.choice_function == Contest.SOCIAL_CHOICE_FUNCTION.APPROVAL:
            self.margin = (tally[self.winner]-tally[self.loser])/self.contest.cards
        elif self.contest.choice_function == Contest.SOCIAL_CHOICE_FUNCTION.SUPERMAJORITY:
            if self.winner == Contest.CANDIDATES.NO_CANDIDATE or self.loser != Contest.CANDIDATES.ALL_OTHERS:
                raise NotImplementedError(f'TO DO: currently only support super-majority with a winner')
            else:
                q = np.sum([tally[c] for c in self.contest.candidates])/self.contest.cards
                p = tally[self.winner]/self.contest.cards
                self.margin = q*(p/self.contest.share_to_win - 1)
        else:
            raise NotImplementedError(f'social choice function {self.contest.choice_function} not supported')


    def make_overstatement(self, overs: float, cvr_list: list=None, use_style: bool=False) -> float:
        '''
        return the numerical value corresponding to an overstatement of `overs` times the assorter upper bound `u`

        **Assumes that the margin has been set.**

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

        '''
        return (1-overs/self.assorter.upper_bound)/(2-self.margin/self.assorter.upper_bound)


    def mvrs_to_data(self, mvr_sample: list=None, cvr_sample: list=None) -> np.array:
        '''
        Process mvrs (and, for comparison audits, cvrs) to create data for the assertion's test 
        and for sample size simulations.
        
        Creates assorter values for the mvrs, or overstatement assorter values using the mvrs and cvrs,
        according to whether the audit uses ballot polling or card-level comparison
        
        The margin should be set before calling this function.
        
        mvr_sample and cvr_sample should be ordered using CVR.prep_comparison_sample() or
           CVR.prep_polling_sample() before calling this routine
        
        Parameters
        ----------
        mvr_sample: list of CVR objects
            corresponding MVRs
        cvr_sample: list of CVR objects
            sampled CVRs
        
        Returns
        -------
        d: np.array
            either assorter values or overstatement assorter values, depending on the audit method
        u: upper bound for the test
        '''
        margin = self.margin
        upper_bound = self.assorter.upper_bound
        con = self.contest
        use_style = con.use_style
        if con.audit_type == Audit.AUDIT_TYPE.CARD_COMPARISON:
            d = np.array(
                [self.overstatement_assorter(mvr_sample[i], cvr_sample[i], use_style=use_style) 
                     for i in range(len(mvr_sample))
                         if ((not use_style) or 
                         (cvr_sample[i].has_contest(con.id) and cvr_sample[i].sample_num <= con.sample_threshold))
                ])
            u = 2/(2-margin/upper_bound)
        elif con.audit_type == Audit.AUDIT_TYPE.POLLING:  # Assume style information is irrelevant
            d = np.array([self.assorter.assort(mvr_sample[i]) for i in range(len(mvr_sample))])
            u = upper_bound
        else:
            raise NotImplementedError(f'audit type {con.audit_type} not implemented')
        return d, u
        
        
    def find_sample_size(
                    self, data: np.array=None, prefix: bool=False, rate_1: float=None, rate_2: float=None,
                    reps: int=None, quantile: float=0.5, seed: int=1234567890) -> int:
        '''
        Estimate sample size needed to reject the null hypothesis that the assorter mean is <=1/2,
        for the specified risk function, given:
            - for comparison audits, the assorter margin and assumptions about the rate of overstatement errors
            - for polling audits, either a set of assorter values, or the assumption that the reported tallies 
              are correct

        If `data is not None`, uses data to make the estimate. There are three strategies:
            1. if `reps is None`, tile the data to make a list of length N
            2. if `reps is not None and not prefix`, sample from the data with replacement to make `reps` lists of
               length N
            3. if `reps is not None and prefix`, start with `data`, then draw N-len(data) times from data with
               replacement to make `reps` lists of length N

        If `data is None`, constructs values from scratch. 
            - For polling audits, values are inferred from the reported tallies. Since contest.tally only reports
              actual candidate totals, not IRV/RAIRE pseudo-candidates, this is not implemented for IRV.
            - For comparison audits, there are two strategies to construct the values:
                1. Systematically interleave small and large values, starting with a small value (`reps is None`)
                2. Sample randomly from a set of such values
            The rate of small values is `rate_1` if `rate_1 is not None`. If `rate is None`, for POLLING audits, gets
            the rate of small values from the margin.
            For Audit.AUDIT_TYPE.POLLING audits, the small values are 0 and the large values are `u`; the rest are 1/2.
            For Audit.AUDIT_TYPE.CARD_COMPARISON audits, the small values are the overstatement assorter for an
            overstatement of `u/2` and the large values are the overstatement assorter for an overstatement of 0.

        This function is for a single assertion.

        **Assumes that self.test.u has been set appropriately for the audit type (polling or comparison).**
        **Thus, for comparison audits, the assorter margin should be be set before calling this function.**

        Parameters
        ----------
        data: np.array
            observations on which to base the calculation. If `data is not None`, uses them in a bootstrap
            approach, rather than simulating errors.
            If `self.contest.audit_type==Audit.POLLING`, the data should be (simulated or actual) values of
            the raw assorter.
            If `self.contest.audit_type==Audit.CARD_COMPARISON`, the data should be (simulated or actual)
            values of the overstatement assorter.
        prefix: bool
            prefix the data, then sample or tile to produce the remaining values
        rate_1: float
            assumed rate of "small" values for simulations (1-vote overstatements). Ignored if `data is not None`
            If `rate_1 is None and self.contest.audit_type==Audit.POLLING` the rate of small values is inferred
            from the margin
        rate_2: float
            assumed rate of 0s for simulations (2-vote overstatements).
        reps: int
            if `reps is None`, builds the data systematically
            if `reps is not None`, performs `reps` simulations to estimate the `quantile` quantile of sample size.
        quantile: float
            if `reps is not None`, quantile of the distribution of sample sizes to return
            if `reps is None`, ignored
        seed: int
            if `reps is not None`, use `seed` as the seed in numpy.random to estimate the quantile

        Returns
        -------
        sample_size: int
            sample size estimated to be sufficient to confirm the outcome if data are generated according to
            the assumptions

        Side effects
        ------------
        sets the sample_size attribute of the assertion

        '''
        assert self.margin > 0, f'Margin {self.margin} is nonpositive'
        if data is not None: 
            sample_size = self.test.sample_size(data, alpha=self.contest.risk_limit, reps=reps,
                                                prefix=prefix, quantile=quantile, seed=seed)
        else:
            '''
            Construct data.
            For POLLING, values are 0, 1/2, and u.
            For CARD_COMPARISON, values are overstatement assorter values corresponding to
            overstatements of 2u (at rate_2), u (at rate_1), or 0.
            '''
            big = (self.assorter.upper_bound if self.contest.audit_type == Audit.AUDIT_TYPE.POLLING \
                  else self.make_overstatement(overs=0))
            small = (0 if self.contest.audit_type == Audit.AUDIT_TYPE.POLLING 
                     else self.make_overstatement(overs=1/2))
            rate_1 = rate_1 if rate_1 is not None else (1-self.margin)/2   # rate of small values
            x = big*np.ones(self.test.N)
            if self.contest.audit_type == Audit.AUDIT_TYPE.POLLING:
                if self.contest.choice_function == Contest.SOCIAL_CHOICE_FUNCTION.IRV:
                    raise NotImplementedError(f'data must be provided to estimate sample sizes for IRV assertions')
                else: # get tally
                    if self.contest.tally:
                        n_0 = self.contest.tally[self.loser]
                        n_big = self.contest.tally[self.winner]
                        n_half = self.test.N - n_0 - n_big
                        x = interleave_values(n_0, n_half, n_big, big=big)
                    else: 
                        raise ValueError(f'contest {self.contest} tally required but not defined')
            elif self.contest.audit_type == Audit.AUDIT_TYPE.CARD_COMPARISON: # comparison audit
                rate_1_i = np.arange(0, self.test.N, step=int(1/rate_1), dtype=int) if rate_1 else []
                rate_2_i = np.arange(0, self.test.N, step=int(1/rate_2), dtype=int) if rate_2 else []
                x[rate_1_i] = small
                x[rate_2_i] = 0
            else:
                raise NotImplementedError(f'audit type {self.contest.audit_type} for contest {self.contest} not implemented')
            sample_size = self.test.sample_size(x, alpha=self.contest.risk_limit, reps=reps,
                                                prefix=prefix, quantile=quantile, seed=seed)
        self.sample_size = sample_size
        return sample_size

    @classmethod
    def interleave_values(
            cls, n_small: int, n_med: int, n_big: int, small: float=0, med: float=1/2, big: float=1):
        r'''
        make an interleaved population of n_s values equal to small, n_m values equal to med, and n_big equal to big
        Start with a small if n_small > 0
        '''
        N = n_small + n_med + n_big
        x = np.zeros(N)
        i_small = 0
        i_med = 0
        i_big = 0
        r_small = 1 if n_small else 0
        r_med = 1 if n_med else 0
        r_big = 1 
        if r_small:   # start with small
            x[0] = small
            i_small = 1
            r_small = (n_small-i_small)/n_small
        elif r_med: # start with 1/2
            x[0] = med
            i_med = 1
            r_med = (n_med-i_med)/n_med
        else:
            x[0] = big
            i_big = 1
            r_big = (n_big-i_big)/n_big
        for i in range(1, N):
            if r_small > r_big:
                if r_med > r_small:
                    x[i] = med
                    i_med += 1
                    r_med = (n_med-i_med)/n_med
                else:
                    x[i] = small
                    i_small += 1
                    r_small = (n_small-i_small)/n_small
            elif r_med > r_big:
                x[i] = med
                i_med += 1
                r_med = (n_med-i_med)/n_med
            else:
                x[i] = big
                i_big += 1
                r_big = (n_big-i_big)/n_big
        return x
                            
    @classmethod
    def make_plurality_assertions(
                                  cls, contest: object=None, winner: list=None, loser: list=None, 
                                  test: callable=None, estim: callable=None, bet: callable=None):
        '''
        Construct assertions that imply the winner(s) got more votes than the loser(s).

        The assertions are that every winner beat every loser: there are
        len(winner)*len(loser) pairwise assertions in all.

        Parameters
        -----------
        contest: instance of Contest
            contest to which the assertions are relevant
        winner: list
            list of identifiers of winning candidate(s)
        loser: list
            list of identifiers of losing candidate(s)

        Returns
        --------
        a dict of Assertions

        '''
        assertions = {}
        test = test if test is not None else contest.test
        estim = estim if estim is not None else contest.estim
        bet = bet if bet is not None else contest.bet
        for winr in winner:
            for losr in loser:
                wl_pair = winr + ' v ' + losr
                _test = NonnegMean(test=test, estim=estim, bet=bet, g=contest.g, u=1, N=contest.cards,
                                       t=1/2, random_order=True)
                assertions[wl_pair] = Assertion(contest, winner=winr, loser=losr,
                                         assorter=Assorter(contest=contest,
                                             assort = lambda c, contest_id=contest.id, winr=winr, losr=losr:
                                                 (CVR.as_vote(c.get_vote_for(contest.id, winr))
                                                 - CVR.as_vote(c.get_vote_for(contest.id, losr))
                                                  + 1)/2,
                                            upper_bound=1),
                                         test=_test)
        return assertions

    @classmethod
    def make_supermajority_assertion(
                                     cls, contest, winner: str=None, loser: list=None,
                                     test: callable=None, estim: callable=None, bet:callable=None):
        '''
        Construct assertion that winner got >= share_to_win \in (0,1) of the valid votes

        **TO DO: This method assumes there was a winner. To audit that there was no winner requires
        flipping things.**

        An equivalent condition is:

        (votes for winner)/(2*share_to_win) + (invalid votes)/2 > 1/2.

        Thus the correctness of a super-majority outcome--where share_to_win >= 1/2--can
        be checked with a single assertion.

        share_to_win < 1/2 might be useful for some social choice functions, including
        primaries where candidates who receive less than some threshold share are
        eliminated.

        A CVR with a mark for more than one candidate in the contest is considered an
        invalid vote.

        Parameters
        -----------
        contest:
            contest object instance to which the assertion applies
        winner:
            identifier of winning candidate
        loser: list
            list of identifiers of losing candidate(s)
        share_to_win: float
            fraction of the valid votes the winner must get to win
        test: instance of NonnegMean
            risk function for the contest
        estim: an estimation method of NonnegMean
            estimator the alpha_mart test uses for the alternative
        bet: method to choose the bet for betting_mart risk function

        Returns
        --------
        a dict containing one Assertion

        '''
        assertions = {}
        wl_pair = winner + ' v ' + Contest.CANDIDATES.ALL_OTHERS
        cands = loser.copy()
        cands.append(winner)
        _test = NonnegMean(test=test, estim=estim, bet=bet, u=1/(2*contest.share_to_win), 
                           N=contest.cards, t=1/2, random_order=True)
        assertions[wl_pair] = Assertion(contest, winner=winner, loser=Contest.CANDIDATES.ALL_OTHERS,
                                 assorter=Assorter(contest=contest,
                                          assort = lambda c, contest_id=contest.id:
                                                CVR.as_vote(c.get_vote_for(contest.id, winner))
                                                      /(2*contest.share_to_win)
                                                if c.has_one_vote(contest.id, cands) else 1/2,
                                          upper_bound=1/(2*contest.share_to_win)), test=_test)
        return assertions

    @classmethod
    def make_assertions_from_json(
                                  cls, contest: object=None, candidates: list=None,
                                  json_assertions: dict=None, test: callable=None,
                                  estim: callable=None, bet: callable=None):
        '''
        dict of Assertion objects from a RAIRE-style json representations of assertions.

        The assertion_type for each assertion must be one of the JSON_ASSERTION_TYPES
        (class constants).

        Parameters
        ----------
        contest: Contest instance
            contest to which the assorter applies
        candidates:
            list of identifiers for all candidates in relevant contest.
        json_assertions:
            Assertions to be tested for the relevant contest.
        test: instance of NonnegMean
            risk function for the contest
        estim: an estimation method of NonnegMean
            estimator the test uses for the alternative

        Returns
        -------
        dict of assertions for each assertion specified in 'json_assertions'.
        '''
        assertions = {}
        for assrtn in json_assertions:
            winr = assrtn['winner']
            losr = assrtn['loser']
            if assrtn['assertion_type'] == cls.WINNER_ONLY:
                # CVR is a vote for the winner only if it has the
                # winner as its first preference
                winner_func = lambda v, contest_id=contest.id, winr=winr: 1 \
                              if v.get_vote_for(contest_id, winr) == 1 else 0

                # CVR is a vote for the loser if they appear and the
                # winner does not, or they appear before the winner
                loser_func = lambda v, contest_id=contest.id, winr=winr, losr=losr: \
                             v.rcv_lfunc_wo(contest_id, winr, losr)

                wl_pair = winr + ' v ' + losr
                _test = NonnegMean(test=test, estim=estim, bet=bet, u=1, N=contest.cards, t=1/2, random_order=True)
                assertions[wl_pair] = Assertion(contest, 
                                                Assorter(contest=contest, winner=winner_func,
                                                   loser=loser_func, upper_bound=1), winner=winr, loser=losr, test=_test)

            elif assrtn['assertion_type'] == cls.IRV_ELIMINATION:
                # Context is that all candidates in 'eliminated' have been
                # eliminated and their votes distributed to later preferences
                elim = [e for e in assrtn['already_eliminated']]
                remn = [c for c in candidates if c not in elim]
                # Identifier for tracking which assertions have been proved
                wl_given = winr + ' v ' + losr + ' elim ' + ' '.join(elim)
                _test = NonnegMean(test=test, estim=estim, bet=bet, u=1, N=contest.cards, t=1/2, random_order=True)
                assertions[wl_given] = Assertion(contest, Assorter(contest=contest,
                                       assort = lambda v, contest_id=contest.id, winner=winr, loser=losr, remn=remn:
                                       ( v.rcv_votefor_cand(contest.id, winner, remn)
                                       - v.rcv_votefor_cand(contest.id, loser, remn) +1)/2,
                                       upper_bound=1), winner=winr, loser=losr, test=_test)
            else:
                raise NotImplemented(f'JSON assertion type {assrtn["assertion_type"]} not implemented.')
        return assertions

    @classmethod
    def make_all_assertions(cls, contests: dict):
        '''
        Construct all the assertions to audit the contests and add the assertions to the contest dict

        Parameters
        ----------
        contests: dict
            dict of Contest objects

        Returns
        -------
        True

        Side Effects
        ------------
        creates assertions and adds the dict of assertions relevant to each contest to the contest
        object's `assertions` attribute

        '''
        for c, con in contests.items():
            scf = con.choice_function
            winrs = con.winner
            losrs = list(set(con.candidates) - set(winrs))
            test = con.test
            estim = con.estim
            bet = con.bet
            if scf == Contest.SOCIAL_CHOICE_FUNCTION.PLURALITY:
                contests[c].assertions = Assertion.make_plurality_assertions(contest=con, winner=winrs, loser=losrs,
                                                                                test=test, estim=estim, bet=bet)
            elif scf == Contest.SOCIAL_CHOICE_FUNCTION.SUPERMAJORITY:
                contests[c].assertions = Assertion.make_supermajority_assertion(contest=con, winner=winrs[0],
                                                    loser=losrs, share_to_win=con.share_to_win,
                                                    test=test, estim=estim, bet=bet)
            elif scf == Contest.SOCIAL_CHOICE_FUNCTION.IRV:
                # Assumption: contests[c].assertion_json yields list assertions in JSON format.
                contests[c].assertions = Assertion.make_assertions_from_json(contest=con,
                                                    candidates=con.candidates,
                                                    json_assertions=con.assertion_json,
                                                    test=test, estim=estim, bet=bet)
            else:
                raise NotImplementedError(f'Social choice function {scf} is not implemented.')
        return True

    @classmethod
    def set_all_margins_from_cvrs(cls, audit: object=None, contests: dict=None, cvr_list: list=None):
        '''
        Find all the assorter margins in a set of Assertions. Updates the dict of dicts of assertions
        and the contest dict.

        Appropriate only if cvrs are available. Otherwise, base margins on the reported results.

        This function is primarily about side-effects on the assertions in the contest dict.

        Parameters
        ----------
        audit: Audit
            information about the audit
        contests: dict of Contest objects
        cvr_list: list
            list of cvr objects

        Returns
        -------
        min_margin: float
            smallest margin in the audit

        Side effects
        ------------
        sets the margin of every assertion
        sets the assertion.test.u for every assertion, according to whether
           `assertion.contest.audit_type==Audit.AUDIT_TYPE.POLLING`
           or `assertion.contest.audit_type==Audit.AUDIT_TYPE.CARD_COMPARISON`
        '''
        min_margin = np.infty
        for c, con in contests.items():
            con.margins = {}
            for a, asn in con.assertions.items():
                asn.set_margin_from_cvrs(audit, cvr_list)
                margin = asn.margin
                con.margins.update({a: margin})
                if con.audit_type==Audit.AUDIT_TYPE.POLLING:
                    u = asn.assorter.upper_bound
                elif con.audit_type==Audit.AUDIT_TYPE.CARD_COMPARISON:
                    u = 2/(2-margin/asn.assorter.upper_bound)
                else:
                    raise NotImplementedError(f'audit type {con.audit_type} not implemented')
                asn.test.u = u
                min_margin = min(min_margin, margin)
        return min_margin


    @classmethod
    def set_p_values(cls, contests: dict, mvr_sample: list, cvr_sample: list=None) -> float :
        '''
        Find the p-value for every assertion and update assertions & contests accordingly

        update p_value, p_history, proved flag, the maximum p-value for each contest.

        Primarily about side-effects.

        Parameters
        ----------
        contests: dict of dicts
            the contest data structure. outer keys are contest identifiers; inner keys are assertions

        mvr_sample: list of CVR objects
            the manually ascertained voter intent from sheets, including entries for phantoms

        cvr_sample: list of CVR objects
            the cvrs for the same sheets, for ballot-level comparison audits
            not needed for polling audits

        Returns
        -------
        p_max: float
            largest p-value for any assertion in any contest

        Side-effects
        ------------
        Sets u for every test for every assertion, according to whether the corresponding audit method
        is Audit.CARD_COMPARISON or Audit.POLLING.
        Sets contest max_p to be the largest P-value of any assertion for that contest
        Updates p_value, p_history, and proved for every assertion

        '''
        if cvr_sample is not None:
            assert len(mvr_sample) == len(cvr_sample), "unequal numbers of cvrs and mvrs"
        p_max = 0
        for c, con in contests.items():
            con.p_values = {}
            con.proved = {}
            contest_max_p = 0
            for a, asn in con.assertions.items():
                d, u = asn.mvrs_to_data(mvr_sample, cvr_sample)
                asn.test.u = u       # set upper bound for the test for each assorter
                asn.p_value, asn.p_history = asn.test.test(d)
                asn.proved = (asn.p_value <= con.risk_limit) or asn.proved
                con.p_values.update({a: asn.p_value})
                con.proved.update({a: asn.proved})
                contest_max_p = np.max([contest_max_p, asn.p_value])
            contests[c].max_p = contest_max_p
            p_max = np.max([p_max, contests[c].max_p])
        return p_max


##########################################################################################
class Assorter:
    '''
    Class for generic Assorter.

    An assorter must either have an `assort` method or both `winner` and `loser` must be defined
    (in which case assort(c) = (winner(c) - loser(c) + 1)/2. )

    Class parameters:
    -----------------
    contest: Contest instance
        the contest to which this Assorter applies

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

    def __init__(
                 self, contest: object=None, assort: callable=None, winner: str=None,
                 loser: str=None, upper_bound: float=1):
        '''
        Constructs an Assorter.

        If assort is defined and callable, it becomes the class instance of assort

        If assort is None but both winner and loser are defined and callable,
           assort is defined to be 1/2 if winner=loser; winner, otherwise


        Parameters
        -----------
        contest: Contest instance
            the contest to which the assorter is relevant
        assort: callable
            maps a dict of votes into [0, upper_bound]
        winner: callable
            maps a pattern into [0, 1]
        loser : callable
            maps a pattern into [0, 1]
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

    def __str__(self):
        '''
        string representation
        '''
        return f'contest_id: {self.contest.id}\nupper bound: {self.upper_bound}, ' +\
               f'winner defined: {callable(self.winner)}, loser defined: {callable(self.loser)}, ' +\
               f'assort defined: {callable(self.assort)}'

    def mean(self, cvr_list: list=None, use_style: bool=True):
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
            filtr = lambda c: c.has_contest(self.contest.id)
        else:
            filtr = lambda c: True
        return np.mean([self.assort(c) for c in cvr_list if filtr(c)])


    def sum(self, cvr_list: list=None, use_style: bool=True):
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
            filtr = lambda c: c.has_contest(self.contest.id)
        else:
            filtr = lambda c: True
        return np.sum([self.assort(c) for c in cvr_list if filtr(c)])

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
            overstatement = self.assort(cvr) \
                            - (self.assort(mvr) if not mvr.phantom else 0)
        elif use_style:
            if cvr.has_contest(self.contest.id):  # make_phantoms() assigns contests but not votes to phantom CVRs
                if cvr.phantom:
                    cvr_assort = 1/2
                else:
                    cvr_assort = self.assort(cvr)
                if mvr.phantom or not mvr.has_contest(self.contest.id):
                    mvr_assort = 0
                else:
                    mvr_assort = self.assort(mvr)
                overstatement = cvr_assort - mvr_assort
            else:
                raise ValueError("Assertion.overstatement: use_style==True but CVR does not contain the contest")
        return overstatement

##########################################################################################
class Contest:
    '''
    Objects and methods for contests.
    '''

    class SOCIAL_CHOICE_FUNCTION:
        '''
        social choice functions
        '''
        SOCIAL_CHOICE_FUNCTIONS = (APPROVAL:= 'APPROVAL',
                                   PLURALITY:= 'PLURALITY',
                                   SUPERMAJORITY:= 'SUPERMAJORITY',
                                   IRV:= 'IRV')
    class CANDIDATES:
        '''
        constants for referring to candidates and candidate groups.

        For example, in a supermajority contest where no candidate is reported to have won,
        the winner is Contest.CANDIDATES.NO_CANDIDATE, and in a supermajority contest in which one
        candidate is reported to have won, the loser (for the assorter) is Contest.CANDIDATES.ALL_OTHERS
        '''
        CANDIDATES = (ALL:= 'ALL',
                      ALL_OTHERS:= 'ALL_OTHERS',
                      WRITE_IN:= 'WRITE_IN',
                      NO_CANDIDATE:= 'NO_CANDIDATE')
    ATTRIBUTES = (
                  'id',
                  'name',
                  'risk_limit',
                  'cards',
                  'choice_function',
                  'n_winners',
                  'share_to_win',
                  'candidates',
                  'winner',
                  'assertion_file',
                  'audit_type',
                  'test',
                  'g',
                  'use_style',
                  'assertions',
                  'tally',
                  'sample_size',
                  'sample_threshold'   
                 )


    def __init__(
                 self,
                 id: object=None,
                 name: str=None,
                 risk_limit: float=0.05,
                 cards: int=0,
                 choice_function: str=SOCIAL_CHOICE_FUNCTION.PLURALITY,
                 n_winners: int=1,
                 share_to_win: float=None,
                 candidates: list=None,
                 winner: list=None,
                 assertion_file: str=None,
                 audit_type: str=Audit.AUDIT_TYPE.CARD_COMPARISON,
                 test: callable=None,
                 g: float=0.1,
                 estim: callable=None,
                 bet: callable=None,
                 use_style: bool=True,
                 assertions: dict=None,
                 tally: dict=None,
                 sample_size: int=None,
                 sample_threshold: float=None):
        self.id = id
        self.name = name
        self.risk_limit = risk_limit
        self.cards = cards
        self.choice_function = choice_function
        self.n_winners = n_winners
        self.share_to_win = share_to_win
        self.candidates = candidates
        self.winner = winner
        self.assertion_file = assertion_file
        self.audit_type = audit_type
        self.test = test
        self.g=g
        self.estim = estim
        self.bet = bet
        self.use_style = use_style
        self.assertions = assertions
        self.tally = tally
        self.sample_size = sample_size
        self.sample_threshold = sample_threshold

    def __str__(self):
        return str(self.__dict__)


    def find_sample_size(
                         self, audit: object=None, mvr_sample: list=None, cvr_sample: list=None, **kwargs) -> int:
        '''
        Estimate the sample size required to confirm the contest at its risk limit.

        This function can be used with or without data, for Audit.AUDIT_TYPE.POLLING and Audit.AUDIT_TYPE.CARD_COMPARISON
        audits.

        The simulations in this implementation are inefficient because the randomization happens separately
        for every assorter, rather than in parallel.

        Parameters
        ----------
        cvrs: list of CVRs
            data (or simulated data) to base the sample size estimates on
        mvrs: list of MVRs (CVR objects)
            manually read votes to base the sample size estimates on, if data are available.

        Returns
        -------
        estimated sample size

        Side effects
        ------------
        sets self.sample_size to the estimated sample size

        '''
        self.sample_size = 0
        for a in self.assertions.values():
            data = None 
            if mvr_sample is not None:  # process the MVRs/CVRs to get data appropriate to each assertion
                data, u = a.mvrs_to_data(mvr_sample, cvr_sample)
            self.sample_size = max(self.sample_size,
                                   a.find_sample_size(data=data, rate_1=audit.error_rate_1, rate_2=audit.error_rate_2,
                                                      reps=audit.reps, quantile=audit.quantile, seed=audit.sim_seed))
        return self.sample_size

    def find_margins_from_tally(self):
        '''
        Use the `Contest.tally` attribute to set the margins of the contest's assorters.

        Appropriate only for the social choice functions
                Contest.SOCIAL_CHOICE_FUNCTION.PLURALITY,
                Contest.SOCIAL_CHOICE_FUNCTION.SUPERMAJORITY,
                Contest.SOCIAL_CHOICE_FUNCTION.APPROVAL


        Parameters
        ----------
        None

        Returns
        -------
        None

        Side effects
        ------------
        sets Assertion.margin for all Assertions in the Contest
        '''
        for a, assn in self.assertions.items():
            assn.find_margin_from_tally()


    @classmethod
    def from_dict(cls, d: dict) -> dict:
        '''
        define a contest objects from a dict containing data for one contest
        '''
        c = Contest()
        c.__dict__.update(d)
        return c

    @classmethod
    def from_dict_of_dicts(cls, d: dict) -> dict:
        '''
        define a dict of contest objects from a dict of dicts, each inner dict containing data for one contest
        '''
        contests = {}
        for di, v in d.items():
            contests[di] = cls.from_dict(v)
            contests[di].id = di
        return contests

    @classmethod
    def from_cvr_list(cls, audit, votes, cards, cvr_list: list=None) -> dict:
        """
        Create a contest dict containing all contests in a cvr_list.
        Every contest is single-winner plurality by default, audited by ballot comparison
        """
        if len(audit.strata) > 1:
            raise NotImplementedError('stratified audits not implemented')
        stratum = next(iter(audit.strata.values()))
        use_style = stratum.use_style
        max_cards = stratum.max_cards
        contest_dict = {}
        for key in votes:
            contest_name = str(key)
            cards_with_contest = cards[key]
            options = np.array(list(votes[key].keys()), dtype = 'str')
            tallies = np.array(list(votes[key].values()))

            reported_winner = options[np.argmax(tallies)]

            contest_dict[contest_name] = {
                'name' : contest_name,
                'cards' : cards_with_contest if use_style else max_cards,
                'choice_function': Contest.SOCIAL_CHOICE_FUNCTION.PLURALITY,
                'n_winners': 1,
                'risk_limit' : 0.05,
                "candidates" : list(options),
                'winner' : [reported_winner],
                'assertion_file': None,
                'audit_type': Audit.AUDIT_TYPE.CARD_COMPARISON,
                'test': NonnegMean.alpha_mart,
                'estim': NonnegMean.optimal_comparison,
                'bet': NonnegMean.fixed_bet
            }
        contests = Contest.from_dict_of_dicts(contest_dict)
        return contests



    @classmethod
    def print_margins(cls, contests: dict=None):
        '''
        print all assorter margins
        '''
        for c, con in contests.items():
            print(f'margins in contest {c}:')
            for a, m in con.margins.items():
                print(f'\tassertion {a}: {m}')
