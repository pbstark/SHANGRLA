import numpy as np
import sys
import pytest
from cryptorandom.cryptorandom import SHA256

from shangrla.core.Audit import Audit, Assertion, Contest, CVR

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
        cvr_dict = [{'id': 1, 'pool': True, 'tally_pool': 1, 'votes': {'AvB': {'Alice':True}, 'CvD': {'Candy':True}}},
                    {'id': 2, 'sample_num': 0.2, 'p': 0.5, 'sampled': True, 
                              'votes': {'AvB': {'Bob':True}, 'CvD': {'Elvis':True, 'Candy':False}}},
                    {'id': 3, 'tally_pool': 'abc', 'votes': {'EvF': {'Bob':1, 'Edie':2}, 'CvD': {'Elvis':False, 'Candy':True}}}]
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
        
        assert cvr_list[0].pool 
        assert cvr_list[0].tally_pool == 1
        assert cvr_list[1].sample_num == 0.2
        assert cvr_list[1].p == 0.5
        assert cvr_list[1].sampled
        assert cvr_list[2].tally_pool == 'abc'

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

    def test_cvr_add_votes(self):
        cvr_dicts = [{'id': 1, 'votes': {'AvB': {}, 'CvD': {'Candy':True}}},
                     {'id': 2, 'votes': {'CvD': {'Elvis':True, 'Candy':False}}}]
        cvr_list = CVR.from_dict(cvr_dicts)
        assert not cvr_list[0].has_contest('QvR')
        assert not cvr_list[1].has_contest('AvB')
        assert cvr_list[0].update_votes({'QvR': {}})
        assert cvr_list[1].get_vote_for('CvD', 'Elvis')
        assert not cvr_list[0].update_votes({'CvD': {'Dan':7}})
        assert cvr_list[1].update_votes({'QvR': {}, 'CvD': {'Dan':7, 'Elvis':False, 'Candy': True}})
        for c in cvr_list:
            assert c.has_contest('QvR')
        assert not cvr_list[0].get_vote_for('QvR', 'Dan')
        assert cvr_list[0].get_vote_for('CvD', 'Dan') == 7
        assert cvr_list[0].get_vote_for('CvD', 'Candy')
        assert not cvr_list[0].get_vote_for('CvD', 'Elvis')
        assert cvr_list[1].get_vote_for('CvD', 'Dan') == 7
        assert cvr_list[1].get_vote_for('CvD', 'Candy')
        assert not cvr_list[1].get_vote_for('CvD', 'Elvis')        

    def test_cvr_pool_contests(self):
        cvr_dicts = [{'id': 1, 'tally_pool': 'a', 'pool': False, 'votes': {'AvB': {}, 'CvD': {'Candy':True}}},
                     {'id': 2, 'tally_pool': 'a', 'pool': False, 'votes': {'CvD': {'Elvis':True, 'Candy':False}, 'EvF': {}}},
                     {'id': 3, 'tally_pool': 'b', 'pool': True, 'votes': {'GvH': {}}},
                     {'id': 4, 'tally_pool': 'b', 'pool': True, 'votes': {'AvB': {}, 'CvD': {'Candy':True}}},
                     
                   ]
        cvr_list = CVR.from_dict(cvr_dicts)
        assert CVR.pool_contests(cvr_list) == {'b':{'AvB', 'CvD', 'GvH'} } 

    def test_add_pool_contests(self):
        cvr_dicts = [{'id': 1, 'tally_pool': 1, 'pool': True, 'votes': {'AvB': {}, 'CvD': {'Candy':True}}},
                     {'id': 2, 'tally_pool': 1, 'pool': True, 'votes': {'CvD': {'Elvis':True, 'Candy':False}, 'EvF': {}}},
                     {'id': 3, 'tally_pool': 1, 'pool': True, 'votes': {'GvH': {}}},
                     {'id': 4, 'tally_pool': 2, 'pool': True, 'votes': {'AvB': {}, 'CvD': {'Candy':True}}},
                     {'id': 5, 'tally_pool': 2, 'pool': True, 'votes': {'CvD': {'Elvis':True, 'Candy':False}, 'EvF': {}}}
                   ]
        cvr_list = CVR.from_dict(cvr_dicts)
        tally_pools = CVR.pool_contests(cvr_list)  
        assert CVR.add_pool_contests(cvr_list, tally_pools)
        for i in range(3):
            assert set(cvr_list[i].votes.keys()) == {'AvB', 'CvD', 'EvF', 'GvH'}  
        for i in range(3,5):
            assert set(cvr_list[i].votes.keys()) == {'AvB', 'CvD', 'EvF'} 
        assert not CVR.add_pool_contests(cvr_list, tally_pools)

    def test_oneaudit_overstatement(self):
        cvr_dicts = [{'id': 1, 'tally_pool': 1, 'pool': True, 'votes': {'AvB': {}, 'CvD': {'Candy':True}}},
                     {'id': 2, 'tally_pool': 1, 'pool': True, 'votes': {'CvD': {'Elvis':True, 'Candy':False}, 'EvF': {}}},
                     {'id': 3, 'tally_pool': 1, 'pool': True, 'votes': {'GvH': {}}},
                     {'id': 4, 'tally_pool': 2, 'pool': True, 'votes': {'AvB': {}, 'CvD': {'Candy':True}}},
                     {'id': 5, 'tally_pool': 2, 'pool': True, 'votes': {'CvD': {'Elvis':True, 'Candy':False}, 'EvF': {}}}
                   ]
        cvr_list = CVR.from_dict(cvr_dicts)
        tally_pools =  CVR.pool_contests(cvr_list)  
        assert CVR.add_pool_contests(cvr_list, tally_pools)
        for i in range(3):
            assert set(cvr_list[i].votes.keys()) == {'AvB', 'CvD', 'EvF', 'GvH'}  
        for i in range(3,5):
            assert set(cvr_list[i].votes.keys()) == {'AvB', 'CvD', 'EvF'} 
        assert not CVR.add_pool_contests(cvr_list, tally_pools)
        # FIX ME! Need to construct assertions
            
    def test_cvr_from_raire(self):
        raire_cvrs = [['1'],
                      ["Contest","339","5","15","16","17","18","45"],
                      ["339","99813_1_1","17"],
                      ["339","99813_1_3","16"],
                      ["339","99813_1_6","18","17","15","16"],
                      ["3","99813_1_6","2"]
                     ]
        c, n = CVR.from_raire(raire_cvrs)
        assert len(c) == 3
        assert c[0].id == "99813_1_1"
        assert c[0].votes == {'339': {'17':1}}
        assert c[2].id == "99813_1_6"
        assert c[2].votes == {'339': {'18':1, '17':2, '15':3, '16':4}, '3': {'2':1}} # merges votes?

    def test_make_phantoms(self):
        audit = Audit.from_dict({'strata': {'stratum_1': {'max_cards':   8,
                                          'use_style':   True,
                                          'replacement': False
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
        for c, cvr in enumerate(cvrs):
            cvr.sample_num = c
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
                                     'sample_size': 4
                                    }
                    }
        con_tests = Contest.from_dict_of_dicts(contests)
        sample_cvr_indices = CVR.consistent_sampling(cvrs, con_tests)
        assert sample_cvr_indices == [0, 1, 2, 5]
        np.testing.assert_approx_equal(con_tests['city_council'].sample_threshold, 2)
        np.testing.assert_approx_equal(con_tests['measure_1'].sample_threshold, 5)

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

    def test_set_card_in_batch_lex(self):
        cvrs = [CVR(id="B-100", votes={"city_council": {"Alice": 1}, "measure_1": {"yes": 1}}, phantom=False,
                   tally_pool="A"),
                CVR(id="B-90", votes={"city_council": {"Bob": 1}, "measure_1": {"yes": 1}}, phantom=False,
                   tally_pool="A"),
                CVR(id="A-1", votes={"city_council": {"Bob": 1}, "measure_1": {"no": 1}}, phantom=False,
                   tally_pool="A"),
                CVR(id="A-20", votes={"city_council": {"Charlie": 1}}, phantom=False,
                   tally_pool="A"),
                CVR(id="C-50", votes={"city_council": {"Doug": 1}}, phantom=False,
                   tally_pool="B"),
                CVR(id="6", votes={"measure_1": {"no": 1}}, phantom=False,
                   tally_pool="B"),
                CVR(id="7-B", votes={"city_council": {"Alice": 1}, "measure_1": {"yes": 1}, "measure_2": {"no":1}},
                    phantom=False, 
                    tally_pool="B"),
                CVR(id="7-A", votes={"measure_1": {"no": 1}, "measure_2": {"yes": 1}}, phantom=False,
                   tally_pool="B")
            ]
        tally_pool = {"A": ""}
        tally_pool_dict = CVR.set_card_in_batch_lex(cvr_list=cvrs)
        assert cvrs[0].card_in_batch == 2
        assert cvrs[1].card_in_batch == 3
        assert cvrs[2].card_in_batch == 0
        assert cvrs[3].card_in_batch == 1
        assert cvrs[4].card_in_batch == 3
        assert cvrs[5].card_in_batch == 0
        assert cvrs[6].card_in_batch == 2
        assert cvrs[7].card_in_batch == 1
        

##########################################################################################
if __name__ == "__main__":
    sys.exit(pytest.main(["-qq"], plugins=None))
