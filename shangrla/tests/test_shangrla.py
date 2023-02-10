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

from Audit import Audit, Assertion, Assorter, Contest, CVR, Stratum
from NonnegMean import NonnegMean
from Dominion import Dominion
from Hart import Hart
import pandas as pd

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
        cvr_dict = [{'id': 1, 'votes': {'AvB': {'Alice':True}, 'CvD': {'Candy':True}}},
                    {'id': 2, 'votes': {'AvB': {'Bob':True}, 'CvD': {'Elvis':True, 'Candy':False}}},
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
        cvr_dict = [{'id': 1, 'votes': {'AvB': {}, 'CvD': {'Candy':True}}},
                    {'id': 2, 'votes': {'CvD': {'Elvis':True, 'Candy':False}}}]
        cvr_list = CVR.from_dict(cvr_dict)
        assert cvr_list[0].has_contest('AvB')
        assert cvr_list[0].has_contest('CvD')
        assert not cvr_list[0].has_contest('EvF')

        assert not cvr_list[1].has_contest('AvB')
        assert cvr_list[1].has_contest('CvD')
        assert not cvr_list[1].has_contest('EvF')

    def test_cvr_from_raire(self):
        raire_cvrs = [['1'],
                      ["Contest","339","5","15","16","17","18","45"],
                      ["339","99813_1_1","17"],
                      ["339","99813_1_3","16"],
                      ["339","99813_1_6","18","17","15","16"],
                      ["3","99813_1_6","2"]
                     ]
        c = CVR.from_raire(raire_cvrs)
        assert len(c) == 3
        assert c[0].id == "99813_1_1"
        assert c[0].votes == {'339': {'17':1}}
        assert c[2].id == "99813_1_6"
        assert c[2].votes == {'339': {'18':1, '17':2, '15':3, '16':4}, '3': {'2':1}} # merges votes?

    def test_make_phantoms(self):
        audit = Audit.from_dict({'strata': {'stratum_1': {'max_cards':   8,
                                          'use_style':   True,
                                          'replacement': False,

                                         }
                                      }})
        contests =  Contest.from_dict_of_dicts({'city_council': {'risk_limit':0.05,
                                     'id': 'city_council',
                                     'cards': None,
                                     'choice_function':'plurality',
                                     'n_winners':3,
                                     'candidates':['Doug','Emily','Frank','Gail','Harry'],
                                     'winner': ['Doug', 'Emily', 'Frank']
                                    },
                     'measure_1':   {'risk_limit':0.05,
                                     'id': 'measure_1',
                                     'cards': 5,
                                     'choice_function':'supermajority',
                                     'share_to_win':2/3,
                                     'n_winners':1,
                                     'candidates':['yes','no'],
                                     'winner': ['yes']
                                    }
                    })
        cvrs = [CVR(id="1", votes={"city_council": {"Alice": 1},     "measure_1": {"yes": 1}}, phantom=False),
                    CVR(id="2", votes={"city_council": {"Bob": 1},   "measure_1": {"yes": 1}}, phantom=False),
                    CVR(id="3", votes={"city_council": {"Bob": 1},   "measure_1": {"no": 1}}, phantom=False),
                    CVR(id="4", votes={"city_council": {"Charlie": 1}}, phantom=False),
                    CVR(id="5", votes={"city_council": {"Doug": 1}}, phantom=False),
                    CVR(id="6", votes={"measure_1": {"no": 1}}, phantom=False)
                ]
        prefix = 'phantom-'

        cvr_list, phantoms = CVR.make_phantoms(audit=audit, contests=contests, cvr_list=cvrs, prefix='phantom-')
        assert len(cvr_list) == 9
        assert phantoms == 3
        assert contests['city_council'].cvrs == 5
        assert contests['measure_1'].cvrs == 4
        assert contests['city_council'].cards == 8
        assert contests['measure_1'].cards == 5
        assert np.sum([c.has_contest('city_council') for c in cvr_list]) == 8, \
                       np.sum([c.has_contest('city_council') for c in cvr_list])
        assert np.sum([c.has_contest('measure_1') for c in cvr_list]) == 5, \
                      np.sum([c.has_contest('measure_1') for c in cvr_list])
        assert np.sum([c.has_contest('city_council') and not c.phantom for c in cvr_list]) ==  5
        assert np.sum([c.has_contest('measure_1') and not c.phantom for c in cvr_list]) == 4

        audit.strata['stratum_1'].use_style = False
        cvr_list, phantoms = CVR.make_phantoms(audit, contests, cvrs, prefix='phantom-')
        assert len(cvr_list) == 8
        assert phantoms == 2
        assert contests['city_council'].cvrs == 5
        assert contests['measure_1'].cvrs == 4
        assert contests['city_council'].cards == 8
        assert contests['measure_1'].cards == 8
        assert np.sum([c.has_contest('city_council') for c in cvr_list]) == 5, \
                       np.sum([c.has_contest('city_council') for c in cvr_list])
        assert np.sum([c.has_contest('measure_1') for c in cvr_list]) == 4, \
                       np.sum([c.has_contest('measure_1') for c in cvr_list])
        assert np.sum([c.has_contest('city_council') and not c.phantom for c in cvr_list]) ==  5
        assert np.sum([c.has_contest('measure_1') and not c.phantom for c in cvr_list]) == 4

    def test_assign_sample_nums(self):
        cvrs = [CVR(id="1", votes={"city_council": {"Alice": 1}, "measure_1": {"yes": 1}}, phantom=False),
                    CVR(id="2", votes={"city_council": {"Bob": 1},   "measure_1": {"yes": 1}}, phantom=False),
                    CVR(id="3", votes={"city_council": {"Bob": 1},   "measure_1": {"no": 1}}, phantom=False),
                    CVR(id="4", votes={"city_council": {"Charlie": 1}}, phantom=False),
                    CVR(id="5", votes={"city_council": {"Doug": 1}}, phantom=False),
                    CVR(id="6", votes={"measure_1": {"no": 1}}, phantom=False)
                ]
        prng = SHA256(1234567890)
        CVR.assign_sample_nums(cvrs, prng)
        assert cvrs[0].sample_num == 100208482908198438057700745423243738999845662853049614266130533283921761365671
        assert cvrs[5].sample_num == 93838330019164869717966768063938259297046489853954854934402443181124696542865

    def test_consistent_sampling(self):
        cvrs = [CVR(id="1", votes={"city_council": {"Alice": 1}, "measure_1": {"yes": 1}}, phantom=False),
                CVR(id="2", votes={"city_council": {"Bob": 1},   "measure_1": {"yes": 1}}, phantom=False),
                CVR(id="3", votes={"city_council": {"Bob": 1},   "measure_1": {"no": 1}}, phantom=False),
                CVR(id="4", votes={"city_council": {"Charlie": 1}}, phantom=False),
                CVR(id="5", votes={"city_council": {"Doug": 1}}, phantom=False),
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
                                     'winner': ['Doug', 'Emily', 'Frank'],
                                     'sample_size': 3
                                    },
                     'measure_1':   {'risk_limit':0.05,
                                     'id': 'measure_1',
                                     'cards': 5,
                                     'choice_function':'supermajority',
                                     'share_to_win':2/3,
                                     'n_winners':1,
                                     'candidates':['yes','no'],
                                     'winner': ['yes'],
                                     'sample_size': 3
                                    }
                    }
        con_tests = Contest.from_dict_of_dicts(contests)
        sample_cvr_indices = CVR.consistent_sampling(cvrs, con_tests)
        assert sample_cvr_indices == [4, 3, 5, 0, 1]

    def test_tabulate_styles(self):
        cvrs = [CVR(id="1", votes={"city_council": {"Alice": 1}, "measure_1": {"yes": 1}}, phantom=False),
                CVR(id="2", votes={"city_council": {"Bob": 1}, "measure_1": {"yes": 1}}, phantom=False),
                CVR(id="3", votes={"city_council": {"Bob": 1}, "measure_1": {"no": 1}}, phantom=False),
                CVR(id="4", votes={"city_council": {"Charlie": 1}}, phantom=False),
                CVR(id="5", votes={"city_council": {"Doug": 1}}, phantom=False),
                CVR(id="6", votes={"measure_1": {"no": 1}}, phantom=False),
                CVR(id="7", votes={"city_council": {"Alice": 1}, "measure_1": {"yes": 1}, "measure_2": {"no":1}},
                          phantom=False),
                CVR(id="8", votes={"measure_1": {"no": 1}, "measure_2": {"yes": 1}}, phantom=False),
                CVR(id="9", votes={"measure_1": {"no": 1}, "measure_3": {"yes": 1}}, phantom=False),
            ]
        t = CVR.tabulate_styles(cvrs)
        assert len(t) == 6
        assert t[frozenset(['measure_1','measure_3'])] == 1
        assert t[frozenset(['city_council','measure_1'])] == 3
        assert t[frozenset(['city_council','measure_1','measure_2'])] == 1


    def test_tabulate_votes(self):
        cvrs = [CVR(id="1", votes={"city_council": {"Alice": 1}, "measure_1": {"yes": 1}}, phantom=False),
                CVR(id="2", votes={"city_council": {"Bob": 1}, "measure_1": {"yes": 1}}, phantom=False),
                CVR(id="3", votes={"city_council": {"Bob": 1}, "measure_1": {"no": 1}}, phantom=False),
                CVR(id="4", votes={"city_council": {"Charlie": 1}}, phantom=False),
                CVR(id="5", votes={"city_council": {"Doug": 1}}, phantom=False),
                CVR(id="6", votes={"measure_1": {"no": 1}}, phantom=False),
                CVR(id="7", votes={"city_council": {"Alice": 1}, "measure_1": {"yes": 1}, "measure_2": {"no":1}},
                          phantom=False),
                CVR(id="8", votes={"measure_1": {"no": 1}, "measure_2": {"yes": 1}}, phantom=False),
                CVR(id="9", votes={"measure_1": {"no": 1}, "measure_3": {"yes": 1}}, phantom=False),
            ]
        d = CVR.tabulate_votes(cvrs)
        assert d['city_council']['Alice'] == 2
        assert d['city_council']['Bob'] == 2
        assert d['city_council']['Doug'] == 1
        assert d['measure_1']['no'] == 4

######################################################################################

class TestAudit:

    def test_from_dict(self):
        d = {
         'seed':           12345678901234567890,
         'cvr_file':       './Data/SFDA2019_PrelimReport12VBMJustDASheets.raire',
         'manifest_file':  './Data/N19 ballot manifest with WH location for RLA Upload VBM 11-14.xlsx',
         'sample_file':    './Data/sample.csv',
         'mvr_file':       './Data/mvr.json',
         'log_file':       './Data/log.json',
         'quantile':       0.8,
         'error_rate_1':   0.001,
         'error_rate_2':   0.0001,
         'reps':           100,
         'strata':         {'stratum_1': {'max_cards':   293555,
                                          'use_style':   True,
                                          'replacement': True,
                                          'audit_type':  Audit.AUDIT_TYPE.BALLOT_COMPARISON,
                                          'test':        NonnegMean.alpha_mart,
                                          'estimator':   NonnegMean.optimal_comparison,
                                          'test_kwargs': {}
                                         }
                           }
        }
        a = Audit.from_dict(d)
        assert a.strata['stratum_1'].max_cards == 293555
        assert a.quantile == 0.8
        assert a.reps == 100



######################################################################################
class TestAssertion:

    con_test_dict = {'id': 'AvB',
                 'name': 'AvB',
                 'risk_limit': 0.05,
                 'cards': 10**4,
                 'choice_function': Contest.SOCIAL_CHOICE_FUNCTION.SUPERMAJORITY,
                 'n_winners': 1,
                 'candidates': 3,
                 'candidates': ['Alice','Bob','Candy'],
                 'winner': ['Alice'],
                 'audit_type': Audit.AUDIT_TYPE.BALLOT_COMPARISON,
                 'share_to_win': 2/3,
                 'test': NonnegMean.alpha_mart,
                 'use_style': True
                }
    con_test = Contest.from_dict(con_test_dict)


    def test_make_plurality_assertions(self):
        winner = ["Alice","Bob"]
        loser = ["Candy","Dan"]
        asrtns = Assertion.make_plurality_assertions(self.con_test, winner, loser)

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
        loser = ['Bob','Candy']
        assn = Assertion.make_supermajority_assertion(contest=self.con_test, winner="Alice",
                                                      loser=loser)

        label = 'Alice v ' + Contest.CANDIDATES.ALL_OTHERS
        votes = CVR.from_vote({"Alice": 1})
        assert assn[label].assorter.assort(votes) == 3/4, "wrong value for vote for winner"

        votes = CVR.from_vote({"Bob": True})
        assert assn[label].assorter.assort(votes) == 0, "wrong value for vote for loser"

        votes = CVR.from_vote({"Dan": True})
        assert assn[label].assorter.assort(votes) == 1/2, "wrong value for invalid vote--Dan"

        votes = CVR.from_vote({"Alice": True, "Bob": True})
        assert assn[label].assorter.assort(votes) == 1/2, "wrong value for invalid vote--Alice & Bob"

        votes = CVR.from_vote({"Alice": False, "Bob": True, "Candy": True})
        assert assn[label].assorter.assort(votes) == 1/2, "wrong value for invalid vote--Bob & Candy"


    def test_rcv_assorter(self):
        import json
        with open('Data/334_361_vbm.json') as fid:
            data = json.load(fid)
            AvB = Contest.from_dict({'id': 'AvB',
                     'name': 'AvB',
                     'risk_limit': 0.05,
                     'cards': 10**4,
                     'choice_function': Contest.SOCIAL_CHOICE_FUNCTION.IRV,
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
        mvr_dict = [{'id': 1, 'votes': {'AvB': {'Alice':True}}},
                    {'id': 2, 'votes': {'AvB': {'Bob':True}}},
                    {'id': 3, 'votes': {'AvB': {}}},
                    {'id': 4, 'votes': {'CvD': {'Elvis':True, 'Candy':False}}},
                    {'id': 'phantom_1', 'votes': {'AvB': {}}, 'phantom': True}]
        mvrs = CVR.from_dict(mvr_dict)

        cvr_dict = [{'id': 1, 'votes': {'AvB': {'Alice':True}}},
                    {'id': 2, 'votes': {'AvB': {'Bob':True}}},
                    {'id': 3, 'votes': {'AvB': {}}},
                    {'id': 4, 'votes': {'CvD': {'Elvis':True}}},
                    {'id': 'phantom_1', 'votes': {'AvB': {}}, 'phantom': True}]
        cvrs = CVR.from_dict(cvr_dict)

        winner = ["Alice"]
        loser = ["Bob"]

        aVb = Assertion(contest=self.con_test, assorter=Assorter(contest_id="AvB",
                        assort = (lambda c, contest_id="AvB", winr="Alice", losr="Bob":
                        ( CVR.as_vote(c.get_vote_for("AvB", winr))
                        - CVR.as_vote(c.get_vote_for("AvB", losr))
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
        mvr_dict = [{'id': 1, 'votes': {'AvB': {'Alice':True}}},
                    {'id': 2, 'votes': {'AvB': {'Bob':True}}},
                    {'id': 3, 'votes': {'AvB': {'Candy':True}}}]
        mvrs = CVR.from_dict(mvr_dict)

        cvr_dict = [{'id': 1, 'votes': {'AvB': {'Alice':True}}},
                    {'id': 2, 'votes': {'AvB': {'Bob':True}}}]
        cvrs = CVR.from_dict(cvr_dict)

        winner = ["Alice"]
        loser = ["Bob", "Candy"]

        aVb = Assertion(contest=self.con_test, assorter=Assorter(contest_id="AvB",
                        assort = (lambda c, contest_id="AvB", winr="Alice", losr="Bob":
                        ( CVR.as_vote(c.get_vote_for("AvB", winr))
                        - CVR.as_vote(c.get_vote_for("AvB", losr))
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


    def test_assorter_sample_size(self):
        # Test Assorter.sample_size using the Kaplan-Wald risk function
        N = int(10**4)
        AvB = Contest.from_dict({'id': 'AvB',
                             'name': 'AvB',
                             'risk_limit': 0.05,
                             'cards': N,
                             'choice_function': Contest.SOCIAL_CHOICE_FUNCTION.PLURALITY,
                             'n_winners': 1,
                             'candidates': ['Alice','Bob', 'Carol'],
                             'winner': ['Alice'],
                             'audit_type': Audit.AUDIT_TYPE.BALLOT_COMPARISON,
                             'test': NonnegMean.kaplan_markov,
                             'tally': {'Alice': 3000, 'Bob': 2000, 'Carol': 1000},
                             'g': 0.1,
                             'use_style': True
                        })
        loser = list(set(AvB.candidates)-set(AvB.winner))
        AvB.assertions = Assertion.make_plurality_assertions(AvB, winner=AvB.winner, loser=loser)
        AvB.find_margins_from_tally()
        for a_id, a in AvB.assertions.items():
            # first test
            rate=0.01
            sam_size1 = a.find_sample_size(data=np.ones(10), prefix=True, rate=rate, reps=None, quantile=0.5, seed=1234567890)
            # Kaplan-Markov martingale is \prod (t+g)/(x+g). For x = [1, 1, ...], sample size should be:
            ss1 = math.ceil(np.log(AvB.risk_limit)/np.log((a.test.t+a.test.g)/(1+a.test.g)))
            assert sam_size1 == ss1
            #
            # second test
            # For "clean", the term is (1/2+g)/(clean+g); for a one-vote overstatement, it is (1/2+g)/(one_over+g).
            sam_size2 = a.find_sample_size(data=None, prefix=True, rate=rate, reps=10**2, quantile=0.5, seed=1234567890)
            clean = 1/(2-a.margin/a.assorter.upper_bound)
            over = clean/2 # corresponds to an overstatement of upper_bound/2, i.e., 1 vote.
            c = (a.test.t+a.test.g)/(clean+a.test.g)
            o = (a.test.t+a.test.g)/(clean/2+a.test.g)
            # the following calculation assumes the audit will terminate before the second overstatement error
            ss2 = math.ceil(np.log(AvB.risk_limit/o)/np.log(c))+1
            assert sam_size2 == ss2
            #
            # third test
            rate = 0.1
            sam_size3 = a.find_sample_size(data=None, prefix=True, rate=rate, reps=10**2, quantile=0.99, seed=1234567890)
            assert sam_size3 > sam_size2

    def test_margin_from_tally(self):
        AvB = Contest.from_dict({'id': 'AvB',
                     'name': 'AvB',
                     'risk_limit': 0.05,
                     'cards': 10**4,
                     'choice_function': Contest.SOCIAL_CHOICE_FUNCTION.PLURALITY,
                     'n_winners': 1,
                     'candidates': ['Alice','Bob','Carol'],
                     'winner': ['Alice'],
                     'audit_type': Audit.AUDIT_TYPE.BALLOT_COMPARISON,
                     'tally': {'Alice': 3000, 'Bob': 2000, 'Carol': 1000},
                     'test': NonnegMean.kaplan_markov,
                     'g': 0.1,
                     'use_style': True
                })
        AvB.assertions = Assertion.make_plurality_assertions(AvB, winner=['Alice'], loser=['Bob','Carol'])
        AvB.find_margins_from_tally()
        assert AvB.assertions['Alice v Bob'].margin == (AvB.tally['Alice'] - AvB.tally['Bob'])/AvB.cards
        assert AvB.assertions['Alice v Carol'].margin == (AvB.tally['Alice'] - AvB.tally['Carol'])/AvB.cards
        tally = {'Alice': 4000, 'Bob': 2000, 'Carol': 1000}
        AvB.assertions['Alice v Carol'].find_margin_from_tally(tally)
        assert AvB.assertions['Alice v Carol'].margin == (tally['Alice'] - tally['Carol'])/AvB.cards


###################################################################################################
class TestContests:

    def test_contests_from_dict_of_dicts(self):
        ids = ['1','2']
        choice_functions = [Contest.SOCIAL_CHOICE_FUNCTION.PLURALITY, Contest.SOCIAL_CHOICE_FUNCTION.SUPERMAJORITY]
        risk_limit = 0.05
        cards = [10000, 20000]
        n_winners = [2, 1]
        candidates = [4, 2]
        audit_type = [Audit.AUDIT_TYPE.POLLING, Audit.AUDIT_TYPE.BALLOT_COMPARISON]
        use_style = True
        atts = ('id','name','risk_limit','cards','choice_function','n_winners','share_to_win','candidates',
                'winner','assertion_file','audit_type','test','use_style')
        contest_dict = {
                 'con_1': {
                 'name': 'contest_1',
                 'risk_limit': 0.05,
                 'cards': 10**4,
                 'choice_function': Contest.SOCIAL_CHOICE_FUNCTION.PLURALITY,
                 'n_winners': 2,
                 'candidates': 5,
                 'candidates': ['alice','bob','carol','dave','erin'],
                 'winner': ['alice','bob'],
                 'audit_type': Audit.AUDIT_TYPE.BALLOT_COMPARISON,
                 'test': NonnegMean.alpha_mart,
                 'use_style': True
                },
                 'con_2': {
                 'name': 'contest_2',
                 'risk_limit': 0.05,
                 'cards': 10**4,
                 'choice_function': Contest.SOCIAL_CHOICE_FUNCTION.SUPERMAJORITY,
                 'n_winners': 1,
                 'candidates': 4,
                 'candidates': ['alice','bob','carol','dave',],
                 'winner': ['alice'],
                 'audit_type': Audit.AUDIT_TYPE.POLLING,
                 'test': NonnegMean.alpha_mart,
                 'use_style': False
                }
                }
        contests = Contest.from_dict_of_dicts(contest_dict)
        for i, c in contests.items():
            assert c.__dict__.get('id') == i
            for att in atts:
                if att != 'id':
                    assert c.__dict__.get(att) == contest_dict[i].get(att)


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

    def test_sample_size(self):
        eta = 0.75
        u = 1
        N = 1000
        t = 1/2
        alpha = 0.05
        prefix = True
        quantile = 0.5
        reps=None
        prefix=False

        test = NonnegMean(test=NonnegMean.alpha_mart,
                              estim=NonnegMean.fixed_alternative_mean,
                              u=u, N=N, t=t, eta=eta)

        x = np.ones(math.floor(N/200))
        sam_size = test.sample_size(x=x, alpha=alpha, reps=reps, prefix=prefix, quantile=quantile)
        np.testing.assert_equal(sam_size, 8) # ((.75/.5)*1+(.25/.5)*0)**8 = 25 > 1/alpha, so sam_size=8
    #
        reps=100
        sam_size = test.sample_size(x=x, alpha=alpha, reps=reps, prefix=prefix, quantile=quantile)
        np.testing.assert_equal(sam_size, 8) # all simulations should give the same answer
    #
        x = 0.75*np.ones(math.floor(N/200))
        sam_size = test.sample_size(x=x, alpha=alpha, reps=reps, prefix=prefix, quantile=quantile)
        np.testing.assert_equal(sam_size, 14) # ((.75/.5)*.75+(.25/.5)*.25)**14 = 22.7 > 1/alpha, so sam_size=14
    #
        g = 0.1
        x = np.ones(math.floor(N/200))

        test = NonnegMean(test=NonnegMean.kaplan_wald,
                              estim=NonnegMean.fixed_alternative_mean,
                              u=u, N=N, t=t, eta=eta, g=g)
        sam_size = test.sample_size(x=x, alpha=alpha, reps=None, prefix=prefix, quantile=quantile)
    #   p-value is \prod ((1-g)*x/t + g), so
        kw_size = math.ceil(math.log(1/alpha)/math.log((1-g)/t + g))
        np.testing.assert_equal(sam_size, kw_size)

        x = 0.75*np.ones(math.floor(N/200))
        sam_size = test.sample_size(x=x, alpha=alpha, reps=None, prefix=prefix, quantile=quantile)
    #   p-value is \prod ((1-g)*x/t + g), so
        kw_size = math.ceil(math.log(1/alpha)/math.log(0.75*(1-g)/t + g))
        np.testing.assert_equal(sam_size, kw_size)

##########################################################################################

class TestDominion:
    def test_sample_from_manifest(self):
        """
        Test the card lookup function
        """
        sample = [1, 99, 100, 101, 121, 200, 201]
        d = [{'Tray #': 1, 'Tabulator Number': 17, 'Batch Number': 1, 'Total Ballots': 100, 'VBMCart.Cart number': 1},
            {'Tray #': 2, 'Tabulator Number': 18, 'Batch Number': 2, 'Total Ballots': 100, 'VBMCart.Cart number': 2},
            {'Tray #': 3, 'Tabulator Number': 19, 'Batch Number': 3, 'Total Ballots': 100, 'VBMCart.Cart number': 3}]
        manifest = pd.DataFrame.from_dict(d)
        manifest['cum_cards'] = manifest['Total Ballots'].cumsum()
        cards, sample_order, mvr_phantoms = Dominion.sample_from_manifest(manifest, sample)
        # cart, tray, tabulator, batch, card in batch, imprint, absolute index
        print(f'{cards=}')
        assert cards[0] == [1, 1, 17, 1, 1, "17-1-1",1]
        assert cards[1] == [1, 1, 17, 1, 99, "17-1-99",99]
        assert cards[2] == [1, 1, 17, 1, 100, "17-1-100",100]
        assert cards[3] == [2, 2, 18, 2, 1, "18-2-1",101]
        assert cards[4] == [2, 2, 18, 2, 21, "18-2-21",121]
        assert cards[5] == [2, 2, 18, 2, 100, "18-2-100",200]
        assert cards[6] == [3, 3, 19, 3, 1, "19-3-1",201]
        assert len(mvr_phantoms) == 0

#TODO: Fix. (XML won't parse)
class TestHart:
    def test_read_cvrs_directory(self):
        cvr_list = Hart.read_cvrs_directory("tests/Data/Hart_CVRs")
        cvr_1 = cvr_list[0]
        cvr_2 = cvr_list[1]
        assert list(cvr_1.votes.keys()) == ["PRESIDENT","GOVERNOR","MAYOR"]
        assert cvr_1.votes['GOVERNOR'] == {}
        assert cvr_2.get_vote_for("MAYOR", "WRITE_IN")
        assert cvr_2.get_vote_for("PRESIDENT", "George Washington")

    def test_read_cvrs_zip(self):
        cvr_list = Hart.read_cvrs_zip("tests/Data/Hart_CVRs.zip")
        cvr_1 = cvr_list[0]
        cvr_2 = cvr_list[1]
        assert list(cvr_1.votes.keys()) == ["PRESIDENT","GOVERNOR","MAYOR"]
        assert cvr_1.votes['GOVERNOR'] == {}
        assert cvr_2.get_vote_for("MAYOR", "WRITE_IN")
        assert cvr_2.get_vote_for("PRESIDENT", "George Washington")

    #TODO: tests for prep_manifest, sample_from_manifest, sample_from_CVRs
    def test_prep_manifest(self):
        #without phantoms
        manifest = pd.read_excel(I)
        max_cards = 1141765
        n_cvrs = 1141765
        manifest, manifest_cards, phantoms = Hart.prep_manifest(manifest, max_cards, n_cvrs)
        assert manifest['Number of Ballots'].astype(int).sum() == max_cards
        assert phantoms == 0
        #with phantoms
        manifest = pd.read_excel("tests/Data/Hart_manifest.xlsx")
        max_cards = 1500000
        manifest, manifest_cards, phantoms = Hart.prep_manifest(manifest, max_cards, n_cvrs)
        assert manifest['Number of Ballots'].astype(int).sum() == max_cards
        assert phantoms == max_cards - n_cvrs

    def test_sample_from_manifest(self):
        cvr_dict = [{'id': "1_1", 'votes': {'AvB': {'Alice':True}}},
                    {'id': "1_2", 'votes': {'AvB': {'Bob':True}}},
                    {'id': "1_3", 'votes': {'AvB': {'Alice':True}}}]
        manifest = pd.DataFrame.from_dict({'Container': ['Mail', 'Mail'], 'Tabulator': [1, 1],\
            'Batch Name': [1, 2], 'Number of Ballots': [1, 2]}, orient = "columns")
        manifest, manifest_cards, phantoms = Hart.prep_manifest(manifest, 3, 3)
        sample_indices = [0,1,2]
        sampled_card_identifiers, sample_order, mvr_phantoms_sample = \
            Hart.sample_from_manifest(manifest, sample_indices)
        assert sampled_card_identifiers[0][4] == '1-1-0'
        assert sampled_card_identifiers[1][4] == '1-2-0'
        assert sampled_card_identifiers[2][4] == '1-2-1'
        assert sample_order['1-1-0']['selection_order'] == 0
        assert sample_order['1-2-0']['selection_order'] == 1
        assert sample_order['1-2-1']['selection_order'] == 2
        assert mvr_phantoms_sample == []



    def test_sample_from_cvrs(self):
        cvr_dict = [{'id': "1_1", 'votes': {'AvB': {'Alice':True}}},
                    {'id': "1_2", 'votes': {'AvB': {'Bob':True}}},
                    {'id': "1_3", 'votes': {'AvB': {'Alice':True}}}]
        manifest = pd.DataFrame.from_dict({'Container': ['Mail', 'Mail'], 'Tabulator': [1, 1],\
            'Batch Name': [1, 2], 'Number of Ballots': [1, 2]}, orient = "columns")
        manifest, manifest_cards, phantoms = Hart.prep_manifest(manifest, 3, 3)
        cvr_list = CVR.from_dict(cvr_dict)
        sampled_cvr_indices = [0,1]
        cards_to_retrieve, sample_order, cvr_sample, mvr_phantoms_sample = \
            Hart.sample_from_cvrs(cvr_list, manifest, sampled_cvr_indices)
        assert len(cards_to_retrieve) == 2
        assert sample_order["1_1"]["selection_order"] == 0
        assert cvr_sample[1] == cvr_list[1]
        assert mvr_phantoms_sample == []


##########################################################################################
if __name__ == "__main__":
    sys.exit(pytest.main(["-qq"], plugins=None))
