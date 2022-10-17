import math
import numpy as np
import scipy as sp
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
from Audit import Audit, Assertion, Assorter, Contest
from NonnegMean import NonnegMean

#######################################################################################################
class TestCVR:
    
    def test_rcv_lfunc_wo(self):
        votes = CVR.from_vote({"Alice": 1, "Bob": 2, "Candy": 3, "Dan": ''})
        assert votes.rcv_lfunc_wo("AvB", "Bob", "Alice") == 1
        assert votes.rcv_lfunc_wo("AvB", "Alice", "Candy") == 0
        assert votes.rcv_lfunc_wo("AvB", "Dan", "Candy") == 1

    def test_rcv_votefor_cand(self):
        votes = CVR.from_vote({"Alice": 1, "Bob": 2, "Candy": 3, "Dan": '', "Ross": 4, "Aaron": 5})
        remaining = ["Bob","Dan","Aaron","Candy"]
        assert votes.rcv_votefor_cand("AvB", "Candy", remaining) == 0
        assert votes.rcv_votefor_cand("AvB", "Alice", remaining) == 0
        assert votes.rcv_votefor_cand("AvB", "Bob", remaining) == 1
        assert votes.rcv_votefor_cand("AvB", "Aaron", remaining) == 0

        remaining = ["Dan","Aaron","Candy"]
        assert votes.rcv_votefor_cand("AvB", "Candy", remaining) == 1
        assert votes.rcv_votefor_cand("AvB", "Alice", remaining) == 0
        assert votes.rcv_votefor_cand("AvB", "Bob", remaining) == 0
        assert votes.rcv_votefor_cand("AvB", "Aaron", remaining) == 0

    def test_cvr_from_dict(self):
        cvr_dict = [{'id': 1, 'votes': {'AvB': {'Alice':True}, 'CvD': {'Candy':True}}},\
                    {'id': 2, 'votes': {'AvB': {'Bob':True}, 'CvD': {'Elvis':True, 'Candy':False}}},\
                    {'id': 3, 'votes': {'EvF': {'Bob':1, 'Edie':2}, 'CvD': {'Elvis':False, 'Candy':True}}}]
        cvr_list = CVR.from_dict(cvr_dict)
        assert len(cvr_list) == 3
        assert cvr_list[0].id == 1
        assert cvr_list[1].id == 2
        assert cvr_list[2].id == 3

        assert cvr_list[0].get_vote_for('AvB', 'Alice') == True
        assert cvr_list[0].get_vote_for('CvD', 'Candy') == True
        assert cvr_list[0].get_vote_for('AvB', 'Bob') == False
        assert cvr_list[0].get_vote_for('EvF', 'Bob') == False

        assert cvr_list[1].get_vote_for('AvB', 'Alice') == False
        assert cvr_list[1].get_vote_for('CvD', 'Candy') == False
        assert cvr_list[1].get_vote_for('CvD', 'Elvis') == True
        assert cvr_list[1].get_vote_for('CvD', 'Candy') == False
        assert cvr_list[1].get_vote_for('CvD', 'Edie') == False
        assert cvr_list[1].get_vote_for('AvB', 'Bob') == True
        assert cvr_list[1].get_vote_for('EvF', 'Bob') == False

        assert cvr_list[2].get_vote_for('AvB', 'Alice') == False
        assert cvr_list[2].get_vote_for('CvD', 'Candy') == True
        assert cvr_list[2].get_vote_for('CvD', 'Edie') == False
        assert cvr_list[2].get_vote_for('AvB', 'Bob') == False
        assert cvr_list[2].get_vote_for('EvF', 'Bob') == 1
        assert cvr_list[2].get_vote_for('EvF', 'Edie') == 2
        assert cvr_list[2].get_vote_for('EvF', 'Alice') == False

    def test_cvr_has_contest(self):
        cvr_dict = [{'id': 1, 'votes': {'AvB': {}, 'CvD': {'Candy':True}}},\
                    {'id': 2, 'votes': {'CvD': {'Elvis':True, 'Candy':False}}}]
        cvr_list = CVR.from_dict(cvr_dict)
        assert cvr_list[0].has_contest('AvB')
        assert cvr_list[0].has_contest('CvD')
        assert not cvr_list[0].has_contest('EvF')

        assert not cvr_list[1].has_contest('AvB')
        assert cvr_list[1].has_contest('CvD')
        assert not cvr_list[1].has_contest('EvF')

    def test_cvr_from_raire(self):
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

    def test_make_phantoms(self):
        contests =  {'city_council': {'risk_limit':0.05,
                                     'id': 'city_council',
                                     'cards': None,
                                     'choice_function':'plurality',
                                     'n_winners':3,
                                     'candidates':['Doug','Emily','Frank','Gail','Harry'],
                                     'reported_winners': ['Doug', 'Emily', 'Frank']
                                    },
                     'measure_1':   {'risk_limit':0.05,
                                     'id': 'measure_1',
                                     'cards': 5,
                                     'choice_function':'supermajority',
                                     'share_to_win':2/3,
                                     'n_winners':1,
                                     'candidates':['yes','no'],
                                     'reported_winners': ['yes']
                                    }
                    }
        cvrs = [CVR(id="1", votes={"city_council": {"Alice": 1},     "measure_1": {"yes": 1}}, phantom=False), \
                    CVR(id="2", votes={"city_council": {"Bob": 1},   "measure_1": {"yes": 1}}, phantom=False), \
                    CVR(id="3", votes={"city_council": {"Bob": 1},   "measure_1": {"no": 1}}, phantom=False), \
                    CVR(id="4", votes={"city_council": {"Charlie": 1}}, phantom=False), \
                    CVR(id="5", votes={"city_council": {"Doug": 1}}, phantom=False), \
                    CVR(id="6", votes={"measure_1": {"no": 1}}, phantom=False)
                ]
        max_cards = 8
        prefix = 'phantom-'

        cvr_list, phantoms = CVR.make_phantoms(max_cards, cvrs, contests, use_style=True, prefix='')
        assert len(cvr_list) == 9
        assert phantoms == 3
        assert contests['city_council']['cvrs'] == 5
        assert contests['measure_1']['cvrs'] == 4
        assert contests['city_council']['cards'] == 8
        assert contests['measure_1']['cards'] == 5
        assert np.sum([c.has_contest('city_council') for c in cvr_list]) == 8, \
                       np.sum([c.has_contest('city_council') for c in cvr_list])
        assert np.sum([c.has_contest('measure_1') for c in cvr_list]) == 5, \
                      np.sum([c.has_contest('measure_1') for c in cvr_list])
        assert np.sum([c.has_contest('city_council') and not c.phantom for c in cvr_list]) ==  5
        assert np.sum([c.has_contest('measure_1') and not c.phantom for c in cvr_list]) == 4

        cvr_list, phantoms = CVR.make_phantoms(max_cards, cvrs, contests, use_style=False, prefix='')
        assert len(cvr_list) == 8
        assert phantoms == 2
        assert contests['city_council']['cvrs'] == 5
        assert contests['measure_1']['cvrs'] == 4
        assert contests['city_council']['cards'] == 8
        assert contests['measure_1']['cards'] == 8
        assert np.sum([c.has_contest('city_council') for c in cvr_list]) == 5, \
                       np.sum([c.has_contest('city_council') for c in cvr_list])
        assert np.sum([c.has_contest('measure_1') for c in cvr_list]) == 4, \
                       np.sum([c.has_contest('measure_1') for c in cvr_list])
        assert np.sum([c.has_contest('city_council') and not c.phantom for c in cvr_list]) ==  5
        assert np.sum([c.has_contest('measure_1') and not c.phantom for c in cvr_list]) == 4

    def test_assign_sample_nums(self):
        cvrs = [CVR(id="1", votes={"city_council": {"Alice": 1}, "measure_1": {"yes": 1}}, phantom=False), \
                    CVR(id="2", votes={"city_council": {"Bob": 1},   "measure_1": {"yes": 1}}, phantom=False), \
                    CVR(id="3", votes={"city_council": {"Bob": 1},   "measure_1": {"no": 1}}, phantom=False), \
                    CVR(id="4", votes={"city_council": {"Charlie": 1}}, phantom=False), \
                    CVR(id="5", votes={"city_council": {"Doug": 1}}, phantom=False), \
                    CVR(id="6", votes={"measure_1": {"no": 1}}, phantom=False)
                ]
        prng = SHA256(1234567890)
        CVR.assign_sample_nums(cvrs,prng)
        assert cvrs[0].sample_num == 100208482908198438057700745423243738999845662853049614266130533283921761365671
        assert cvrs[5].sample_num == 93838330019164869717966768063938259297046489853954854934402443181124696542865

    def test_consistent_sampling(self):
        cvrs = [CVR(id="1", votes={"city_council": {"Alice": 1}, "measure_1": {"yes": 1}}, phantom=False), \
                CVR(id="2", votes={"city_council": {"Bob": 1},   "measure_1": {"yes": 1}}, phantom=False), \
                CVR(id="3", votes={"city_council": {"Bob": 1},   "measure_1": {"no": 1}}, phantom=False), \
                CVR(id="4", votes={"city_council": {"Charlie": 1}}, phantom=False), \
                CVR(id="5", votes={"city_council": {"Doug": 1}}, phantom=False), \
                CVR(id="6", votes={"measure_1": {"no": 1}}, phantom=False)
                ]
        prng = SHA256(1234567890)
        CVR.assign_sample_nums(cvrs, prng)
        contests = {'city_council': {'risk_limit':0.05,
                                     'id': 'city_council',
                                     'cards': None,
                                     'choice_function':'plurality',
                                     'n_winners':3,
                                     'candidates':['Doug','Emily','Frank','Gail','Harry'],
                                     'reported_winners': ['Doug', 'Emily', 'Frank'],
                                     'sample_size': 3
                                    },
                     'measure_1':   {'risk_limit':0.05,
                                     'id': 'measure_1',
                                     'cards': 5,
                                     'choice_function':'supermajority',
                                     'share_to_win':2/3,
                                     'n_winners':1,
                                     'candidates':['yes','no'],
                                     'reported_winners': ['yes'],
                                     'sample_size': 3
                                    }
                    }
        con_tests = Contest.from_dict_of_dicts(contests)
        sample_cvr_indices = CVR.consistent_sampling(cvrs, con_tests)
        assert sample_cvr_indices == [4, 3, 5, 0, 1]

######################################################################################
class TestAssertion:
    
    con_test_dict = {'id': 'AvB',
                 'name': 'AvB',
                 'risk_limit': 0.05,
                 'cards': 10**4,
                 'choice_function': Audit.SOCIAL_CHOICE_FUNCTION.SUPERMAJORITY,
                 'n_winners': 1,
                 'candidates': 3,
                 'candidates': ['Alice','Bob','Candy'],
                 'reported_winners': ['Alice'],
                 'audit_type': Audit.AUDIT_TYPE.BALLOT_COMPARISON,
                 'share_to_win': 2/3,
                 'test': NonnegMean.alpha_mart,
                 'use_style': True
                }
    con_test = Contest.from_dict(con_test_dict)


    def test_make_plurality_assertions(self):
        winners = ["Alice","Bob"]
        losers = ["Candy","Dan"]
        asrtns = Assertion.make_plurality_assertions(self.con_test, winners, losers)

        assert asrtns['Alice v Candy'].assorter.assort(CVR.from_vote({'Alice': 1})) == 1, \
               f"{asrtns['Alice v Candy'].assorter.assort(CVR.from_vote({'Alice': 1}))=}"
        assert asrtns['Alice v Candy'].assorter.assort(CVR.from_vote({'Bob': 1})) == 1/2
        assert asrtns['Alice v Candy'].assorter.assort(CVR.from_vote({'Candy': 1})) == 0
        assert asrtns['Alice v Candy'].assorter.assort(CVR.from_vote({'Dan': 1})) == 1/2

        assert asrtns['Alice v Dan'].assorter.assort(CVR.from_vote({'Alice': 1})) == 1
        assert asrtns['Alice v Dan'].assorter.assort(CVR.from_vote({'Bob': 1})) == 1/2
        assert asrtns['Alice v Dan'].assorter.assort(CVR.from_vote({'Candy': 1})) == 1/2
        assert asrtns['Alice v Dan'].assorter.assort(CVR.from_vote({'Dan': 1})) == 0

        assert asrtns['Bob v Candy'].assorter.assort(CVR.from_vote({'Alice': 1})) == 1/2
        assert asrtns['Bob v Candy'].assorter.assort(CVR.from_vote({'Bob': 1})) == 1
        assert asrtns['Bob v Candy'].assorter.assort(CVR.from_vote({'Candy': 1})) == 0
        assert asrtns['Bob v Candy'].assorter.assort(CVR.from_vote({'Dan': 1})) == 1/2

        assert asrtns['Bob v Dan'].assorter.assort(CVR.from_vote({'Alice': 1})) == 1/2
        assert asrtns['Bob v Dan'].assorter.assort(CVR.from_vote({'Bob': 1})) == 1
        assert asrtns['Bob v Dan'].assorter.assort(CVR.from_vote({'Candy': 1})) == 1/2
        assert asrtns['Bob v Dan'].assorter.assort(CVR.from_vote({'Dan': 1})) == 0

    def test_supermajority_assorter(self):
        losers = ['Bob','Candy']
        assn = Assertion.make_supermajority_assertion(contest=self.con_test, winner="Alice", 
                                                      losers=losers)

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


    def test_rcv_assorter(self):
        import json
        with open('Data/334_361_vbm.json') as fid:
            data = json.load(fid)
            AvB = Contest.from_dict({'id': 'AvB',
                     'name': 'AvB',
                     'risk_limit': 0.05,
                     'cards': 10**4,
                     'choice_function': Audit.SOCIAL_CHOICE_FUNCTION.IRV,
                     'n_winners': 1,
                     'test': NonnegMean.alpha_mart,
                     'use_style': True
                })
            assertions = {}
            for audit in data['audits']:
                cands = [audit['winner']]
                for elim in audit['eliminated']:
                    cands.append(elim)
                all_assertions = Assertion.make_assertions_from_json(contest=AvB, candidates=cands, 
                                                                     json_assertions=audit['assertions'])
                assertions[audit['contest']] = all_assertions
            
            # winner only assertion
            assorter = assertions['334']['5 v 47'].assorter
            
            votes = CVR.from_vote({'5': 1, '47': 2})
            assert assorter.assort(votes) == 1, f'{assorter.assort(votes)=}'

            votes = CVR.from_vote({'47': 1, '5': 2})
            assert assorter.assort(votes) == 0, f'{assorter.assort(votes)=}'

            votes = CVR.from_vote({'3': 1, '6': 2})
            assert assorter.assort(votes) == 0.5, f'{assorter.assort(votes)=}'

            votes = CVR.from_vote({'3': 1, '47': 2})
            assert assorter.assort(votes) == 0, f'{assorter.assort(votes)=}'

            votes = CVR.from_vote({'3': 1, '5': 2})
            assert assorter.assort(votes) == 0.5, f'{assorter.assort(votes)=}'
            
            # elimination assertion
            assorter = assertions['334']['5 v 3 elim 1 6 47'].assorter
                  
            votes = CVR.from_vote({'5': 1, '47': 2})
            assert assorter.assort(votes) == 1, f'{assorter.assort(votes)=}'

            votes = CVR.from_vote({'47': 1, '5': 2})
            assert assorter.assort(votes) == 1, f'{assorter.assort(votes)=}'

            votes = CVR.from_vote({'6': 1, '1': 2, '3': 3, '5': 4})
            assert assorter.assort(votes) == 0, f'{assorter.assort(votes)=}'

            votes = CVR.from_vote({'3': 1, '47': 2})
            assert assorter.assort(votes) == 0, f'{assorter.assort(votes)=}'

            votes = CVR.from_vote({})
            assert assorter.assort(votes) == 0.5, f'{assorter.assort(votes)=}'

            votes = CVR.from_vote({'6': 1, '47': 2})
            assert assorter.assort(votes) == 0.5, f'{assorter.assort(votes)=}'

            votes = CVR.from_vote({'6': 1, '47': 2, '5': 3})
            assert assorter.assort(votes) == 1, f'{assorter.assort(votes)=}'

            # winner-only assertion
            assorter = assertions['361']['28 v 50'].assorter
            
            votes = CVR.from_vote({'28': 1, '50': 2})
            assert assorter.assort(votes) == 1, f'{assorter.assort(votes)=}'
            
            votes = CVR.from_vote({'28': 1})
            assert assorter.assort(votes) == 1, f'{assorter.assort(votes)=}'
            
            votes = CVR.from_vote({'50': 1})
            assert assorter.assort(votes) == 0, f'{assorter.assort(votes)=}'

            votes = CVR.from_vote({'27': 1, '28': 2})
            assert assorter.assort(votes) == 0.5, f'{assorter.assort(votes)=}'

            votes = CVR.from_vote({'50': 1, '28': 2})
            assert assorter.assort(votes) == 0, f'{assorter.assort(votes)=}'

            votes = CVR.from_vote({'27': 1, '26': 2})
            assert assorter.assort(votes) == 0.5, f'{assorter.assort(votes)=}'

            votes = CVR.from_vote({})
            assert assorter.assort(votes) == 0.5, f'{assorter.assort(votes)=}'

            # elimination assertion
            assorter = assertions['361']['27 v 26 elim 28 50'].assorter
            
            votes = CVR.from_vote({'27': 1})
            assert assorter.assort(votes) == 1, f'{assorter.assort(votes)=}'

            votes = CVR.from_vote({'50': 1, '27': 2})
            assert assorter.assort(votes) == 1, f'{assorter.assort(votes)=}'

            votes = CVR.from_vote({'28': 1, '50': 2, '27': 3})
            assert assorter.assort(votes) == 1, f'{assorter.assort(votes)=}'

            votes = CVR.from_vote({'28': 1, '27': 2, '50': 3})
            assert assorter.assort(votes) == 1, f'{assorter.assort(votes)=}'

            votes = CVR.from_vote({'26': 1})
            assert assorter.assort(votes) == 0, f'{assorter.assort(votes)=}'

            votes = CVR.from_vote({'50': 1, '26': 2})
            assert assorter.assort(votes) == 0, f'{assorter.assort(votes)=}'

            votes = CVR.from_vote({'28': 1, '50': 2, '26': 3})
            assert assorter.assort(votes) == 0, f'{assorter.assort(votes)=}'

            votes = CVR.from_vote({'28': 1, '26': 2, '50': 3})
            assert assorter.assort(votes) == 0, f'{assorter.assort(votes)=}'

            votes = CVR.from_vote({'50': 1})
            assert assorter.assort(votes) == 0.5, f'{assorter.assort(votes)=}'
            
            votes = CVR.from_vote({})
            assert assorter.assort(votes) == 0.5, f'{assorter.assort(votes)=}'

            votes = CVR.from_vote({'50': 1, '28': 2})
            assert assorter.assort(votes) == 0.5, f'{assorter.assort(votes)=}'

            votes = CVR.from_vote({'28': 1, '50': 2})
            assert assorter.assort(votes) == 0.5, f'{assorter.assort(votes)=}'


    def test_overstatement(self):
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

        aVb = Assertion(contest=self.con_test, assorter=Assorter(contest_id="AvB", \
                        assort = (lambda c, contest_id="AvB", winr="Alice", losr="Bob":\
                        ( CVR.as_vote(c.get_vote_for("AvB", winr)) \
                        - CVR.as_vote(c.get_vote_for("AvB", losr)) \
                        + 1)/2), upper_bound=1))
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


    def test_overstatement_assorter(self):
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

        aVb = Assertion(contest=self.con_test, assorter=Assorter(contest_id="AvB", \
                        assort = (lambda c, contest_id="AvB", winr="Alice", losr="Bob":\
                        ( CVR.as_vote(c.get_vote_for("AvB", winr)) \
                        - CVR.as_vote(c.get_vote_for("AvB", losr)) \
                        + 1)/2), upper_bound=1))
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


    def test_assorter_mean(self):
        pass # [FIX ME]


    def test_assorter_sample_size(self):
        # Test Assorter.sample_size using the Kaplan-Wald risk function
        
        rate = 0.01
        N = int(10**4)
        margin = 0.1
        upper_bound = 1
        u = 2/(2-margin/upper_bound)
        m = (1 - rate*upper_bound/margin)/(2*upper_bound/margin - 1)
        one_over = 1/3.8 # 0.5/(2-margin)
        clean = 1/1.9    # 1/(2-margin)

        AvB = Contest.from_dict({'id': 'AvB',
                     'name': 'AvB',
                     'risk_limit': 0.05,
                     'cards': 10**4,
                     'choice_function': Audit.SOCIAL_CHOICE_FUNCTION.PLURALITY,
                     'n_winners': 1,
                     'candidates': ['Alice','Bob'],
                     'test': NonnegMean.kaplan_markov,
                     'g': 0.1,
                     'use_style': True
                })
        assertions = Assertion.make_plurality_assertions(AvB, winners=['Alice'], losers=['Bob'])
        # first test
        bias_up = False
        for a_id, a in assertions.items():
            a.margin = margin
            sam_size = a.find_sample_size(data=None, prefix=True, rate=rate, reps=None, quantile=0.5, seed=1234567890)
            sam_size_1 = 72 # (1/1.9)*(2/1.9)**71 = 20.08
            np.testing.assert_almost_equal(sam_size, sam_size_1)
            # 2nd test
            sam_size = a.find_sample_size(data=None, prefix=True, rate=rate, reps=10**3, quantile=0.5, seed=1234567890)
            np.testing.assert_array_less(sam_size, sam_size_1+1) # crude test, but ballpark
    
###################################################################################################
class TestContests:
    def test_contests_from_dict_of_dicts(self):
        ids = ['1','2']
        choice_functions = [Audit.SOCIAL_CHOICE_FUNCTION.PLURALITY, Audit.SOCIAL_CHOICE_FUNCTION.SUPERMAJORITY]
        risk_limit = 0.05
        cards = [10000, 20000]
        n_winners = [2, 1]
        candidates = [4, 2]
        audit_type = [Audit.AUDIT_TYPE.POLLING, Audit.AUDIT_TYPE.BALLOT_COMPARISON]
        use_style = True
        atts = ('id','name','risk_limit','cards','choice_function','n_winners','share_to_win','candidates',
                'reported_winners','assertion_file','audit_type','test','use_style')
        contest_dict = {
                 'con_1': {'id': '1',
                 'name': 'contest_1',
                 'risk_limit': 0.05,
                 'cards': 10**4,
                 'choice_function': Audit.SOCIAL_CHOICE_FUNCTION.PLURALITY,
                 'n_winners': 2,
                 'candidates': 5,
                 'candidates': ['alice','bob','carol','dave','erin'],
                 'reported_winners': ['alice','bob'],
                 'audit_type': Audit.AUDIT_TYPE.BALLOT_COMPARISON,
                 'test': NonnegMean.alpha_mart,
                 'use_style': True
                },
                 'con_2': {'id': '2',
                 'name': 'contest_2',
                 'risk_limit': 0.05,
                 'cards': 10**4,
                 'choice_function': Audit.SOCIAL_CHOICE_FUNCTION.SUPERMAJORITY,
                 'n_winners': 1,
                 'candidates': 4,
                 'candidates': ['alice','bob','carol','dave',],
                 'reported_winners': ['alice'],
                 'audit_type': Audit.AUDIT_TYPE.POLLING,
                 'test': NonnegMean.alpha_mart,
                 'use_style': False    
                }
                }
        contests = Contest.from_dict_of_dicts(contest_dict)
        for c in contests:
            for att in atts:
                assert contests[c].__dict__.get(att) == contest_dict[c].get(att)

##########################################################################################
class TestNonnegMean:

    def test_alpha_mart(self):
        eps = 0.0001  # Generic small value

        # When all the items are 1/2, estimated p for a mean of 1/2 should be 1.
        s = np.ones(5)/2
        test = NonnegMean(N=int(10**6))
        np.testing.assert_almost_equal(test.alpha_mart(s)[0],1.0)
        test.t = eps
        np.testing.assert_array_less(test.alpha_mart(s)[1][1:],[eps]*(len(s)-1))

        s = [0.6,0.8,1.0,1.2,1.4]
        test.u=2
        np.testing.assert_array_less(test.alpha_mart(s)[1][1:],[eps]*(len(s)-1))

        s1 = [1, 0, 1, 1, 0, 0, 1]
        test.u=1
        test.N = 7
        test.t = 3/7
        alpha_mart1 = test.alpha_mart(s1)[1]
        # p-values should be big until the last, which should be 0
        print(f'{alpha_mart1=}')
        assert(not any(np.isnan(alpha_mart1)))
        assert(alpha_mart1[-1] == 0)

        s2 = [1, 0, 1, 1, 0, 0, 0]
        alpha_mart2 = test.alpha_mart(s2)[1]
        # Since s1 and s2 only differ in the last observation,
        # the resulting martingales should be identical up to the next-to-last.
        # Final entry in alpha_mart2 should be 1
        assert(all(np.equal(alpha_mart2[0:(len(alpha_mart2)-1)],
                            alpha_mart1[0:(len(alpha_mart1)-1)])))
        print(f'{alpha_mart2=}')

    def test_shrink_trunc(self):
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
        test_inf = NonnegMean(N=np.inf, t=t, u=u, d=d, f=f)
        test_fin = NonnegMean(          t=t, u=u, d=d, f=f)
        for eta in etas:
            c = (eta-t)/2
            test_inf.c = c
            test_inf.eta = eta
            test_fin.c=c
            test_fin.eta=eta
            for x in v:
                N = len(x)
                test_fin.N = N
                xinf = test_inf.shrink_trunc(x)
                xfin = test_fin.shrink_trunc(x)
                yinf = np.zeros(N)
                yfin = np.zeros(N)
                for j in range(1,N+1):
                    est = (d*eta + Sj(x,j))/(d+j-1)
                    most = u*(1-np.finfo(float).eps)
                    yinf[j-1] = np.minimum(np.maximum(t+epsj(c,d,j), est), most)
                    yfin[j-1] = np.minimum(np.maximum(tj(N,t,x,j)+epsj(c,d,j), est), most)
                np.testing.assert_allclose(xinf, yinf)
                np.testing.assert_allclose(xfin, yfin)

    def test_kaplan_markov(self):
        s = np.ones(5)
        test = NonnegMean(u=1, N=np.inf, t=1/2)
        np.testing.assert_almost_equal(test.kaplan_markov(s)[0], 2**-5)
        s = np.array([1, 1, 1, 1, 1, 0])
        test.g=0.1
        np.testing.assert_almost_equal(test.kaplan_markov(s)[0],(1.1/.6)**-5)
        test.random_order = False
        np.testing.assert_almost_equal(test.kaplan_markov(s)[0],(1.1/.6)**-5 * .6/.1)
        s = np.array([1, -1])
        try:
            test.kaplan_markov(s)
        except ValueError:
            pass
        else:
            raise AssertionError

    def test_kaplan_wald(self):
        s = np.ones(5)
        test = NonnegMean()
        np.testing.assert_almost_equal(test.kaplan_wald(s)[0], 2**-5)
        s = np.array([1, 1, 1, 1, 1, 0])
        test.g = 0.1
        np.testing.assert_almost_equal(test.kaplan_wald(s)[0], (1.9)**-5)
        test.random_order = False
        np.testing.assert_almost_equal(test.kaplan_wald(s)[0], (1.9)**-5 * 10)
        s = np.array([1, -1])
        try:
            test.kaplan_wald(s)
        except ValueError:
            pass
        else:
            raise AssertionError

    def test_kaplan_kolmogorov(self):
        # NEEDS WORK! This just prints; it doesn't really test anything.
        N = 100
        x = np.ones(10)
        test = NonnegMean(N=N)
        p1 = test.kaplan_kolmogorov(x)
        x = np.zeros(10)
        test.g = 0.1
        p2 = test.kaplan_kolmogorov(x)
        print(f'kaplan_kolmogorov: {p1} {p2}')

    def test_sample_size(self):
        eta = 0.75
        u = 1
        N = 1000
        x = np.ones(math.floor(N/200))
        t = 1/2
        risk_limit = 0.05
        prefix = True
        quantile = 0.5
        reps=None
        prefix=False
        test = NonnegMean(test=NonnegMean.alpha_mart, 
                              estim=NonnegMean.fixed_alternative_mean, 
                              u=u, N=N, t=t, eta=eta)
        sam_size = test.sample_size(x=x, risk_limit=risk_limit, reps=reps, prefix=prefix, quantile=quantile)
        np.testing.assert_equal(sam_size, 8) # ((.75/.5)*1+(.25/.5)*0)**8 = 25 > 1/alpha, so sam_size=8
    #    
        reps=100
        sam_size = test.sample_size(x=x, risk_limit=risk_limit, reps=reps, prefix=prefix, quantile=quantile)
        np.testing.assert_equal(sam_size, 8) # all simulations should give the same answer
    #
        x = 0.75*np.ones(math.floor(N/200))
        sam_size = test.sample_size(x=x, risk_limit=risk_limit, reps=reps, prefix=prefix, quantile=quantile)
        np.testing.assert_equal(sam_size, 14) # ((.75/.5)*.75+(.25/.5)*.25)**14 = 22.7 > 1/alpha, so sam_size=14    


##########################################################################################    
if __name__ == "__main__":
    sys.exit(pytest.main(["-qq"], plugins=None))
