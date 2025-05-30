import math
import numpy as np
import sys
import pytest
from collections import defaultdict

from shangrla.core.Audit import Audit, Assertion, Assorter, Contest, CVR
from shangrla.core.NonnegMean import NonnegMean


######################################################################################
class TestAssertion:
    

    def test_min_p(self):
        asrtn1 = Assertion(p_value = 0.1, p_history = [1, 0.5, 0.1, 0.01, 0.1])
        asrtn2 = Assertion(p_value = 0.05, p_history = [0.05])
        assert Assertion.min_p(asrtn1) == 0.01
        assert Assertion.min_p(asrtn2) == 0.05

    #what's the right way to write a unit test for this?
    #define a bare-minimum Assertion / Assorter and CVR list and then call margin on them?
    def test_margin(self, raw_AvB_asrtn, plur_cvr_list):
        assert Assertion.margin(raw_AvB_asrtn, plur_cvr_list) == 0.5

    # is it weird that margin calling signature involves cvr_list,
    # but not so for overstatement_assorter_margin
    def test_overstatement_assorter_margin(self, plur_con_test):
        AvB_asrtn = Assertion(
            contest = plur_con_test,
            winner = "Alice",
            loser = "Bob",
            assorter = Assorter(
                contest = plur_con_test,
                assort = lambda c:
                             (CVR.as_vote(c.get_vote_for("AvB", "Alice"))
                             - CVR.as_vote(c.get_vote_for("AvB", "Bob"))
                              + 1)/2,
                upper_bound = 1
            ),
            margin = 0.5
        )
        assert Assertion.overstatement_assorter_margin(AvB_asrtn) == 1 / 3

    def test_overstatement_assorter_mean(self, plur_con_test):
        AvB_asrtn = Assertion(
            contest = plur_con_test,
            winner = "Alice",
            loser = "Bob",
            assorter = Assorter(
                contest = plur_con_test,
                assort = lambda c:
                             (CVR.as_vote(c.get_vote_for("AvB", "Alice"))
                             - CVR.as_vote(c.get_vote_for("AvB", "Bob"))
                              + 1)/2,
                upper_bound = 1
            ),
            margin = 0.5
        )
        assert Assertion.overstatement_assorter_mean(AvB_asrtn) == 1/1.5
        assert Assertion.overstatement_assorter_mean(AvB_asrtn, error_rate_1 = 0.5) == 0.5
        assert Assertion.overstatement_assorter_mean(AvB_asrtn, error_rate_2 = 0.25) == 0.5
        assert Assertion.overstatement_assorter_mean(AvB_asrtn, error_rate_1 = 0.25, error_rate_2 = 0.25) == \
            (1 - 0.125 - 0.25)/(2-0.5)

    def test_set_margin_from_cvrs(self, raw_AvB_asrtn, comparison_audit, plur_cvr_list):
        raw_AvB_asrtn.set_margin_from_cvrs(comparison_audit, plur_cvr_list)
        assert raw_AvB_asrtn.margin == 0.5


    def test_make_plurality_assertions(self, con_test):
        winner = ["Alice","Bob"]
        loser = ["Candy","Dan"]
        asrtns = Assertion.make_plurality_assertions(con_test, winner, loser)

        #these test Assorter.assort()
        assert asrtns['Alice v Candy'].assorter.assort(CVR.from_vote({'Alice': 1})) == 1
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

        #this tests Assertions.assort()
        # assert asrtns['Alice v Dan'].assort(CVR.from_vote({'Alice': 1})) == 1
        # assert asrtns['Alice v Dan'].assort(CVR.from_vote({'Bob': 1})) == 1/2
        # assert asrtns['Alice v Dan'].assort(CVR.from_vote({'Candy': 1})) == 1/2
        # assert asrtns['Alice v Dan'].assort(CVR.from_vote({'Dan': 1})) == 0

    def test_supermajority_assorter(self, con_test):
        loser = ['Bob','Candy']
        assn = Assertion.make_supermajority_assertion(contest=con_test, winner="Alice",
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


    def test_rcv_assorter(self, AvB_IRV):
        import json
        with open('./tests/core/data/334_361_vbm.json') as fid:
            data = json.load(fid)
            assertions = {}
            for audit in data['audits']:
                cands = [audit['winner']]
                for elim in audit['eliminated']:
                    cands.append(elim)
                all_assertions = Assertion.make_assertions_from_json(contest=AvB_IRV, candidates=cands,
                                                                     json_assertions=audit['assertions'])
                assertions[audit['contest']] = all_assertions

            # winner only assertion
            assorter = assertions['334']['5 v 47'].assorter
            print(f'{str(assorter)=}')
            votes = CVR.from_vote({'5': 1, '47': 2}, id='AvB_IRV', contest_id='AvB_IRV',)
            assert assorter.assort(votes) == 1, f'{assorter.assort(votes)=}'

            votes = CVR.from_vote({'47': 1, '5': 2}, id='AvB_IRV', contest_id='AvB_IRV',)
            assert assorter.assort(votes) == 0, f'{assorter.assort(votes)=}'

            votes = CVR.from_vote({'3': 1, '6': 2}, id='AvB_IRV', contest_id='AvB_IRV',)
            assert assorter.assort(votes) == 0.5, f'{assorter.assort(votes)=}'

            votes = CVR.from_vote({'3': 1, '47': 2}, id='AvB_IRV', contest_id='AvB_IRV',)
            assert assorter.assort(votes) == 0, f'{assorter.assort(votes)=}'

            votes = CVR.from_vote({'3': 1, '5': 2}, id='AvB_IRV', contest_id='AvB_IRV',)
            assert assorter.assort(votes) == 0.5, f'{assorter.assort(votes)=}'

            # elimination assertion
            assorter = assertions['334']['5 v 3 elim 1 6 47'].assorter

            votes = CVR.from_vote({'5': 1, '47': 2}, id='AvB_IRV', contest_id='AvB_IRV',)
            assert assorter.assort(votes) == 1, f'{assorter.assort(votes)=}'

            votes = CVR.from_vote({'47': 1, '5': 2}, id='AvB_IRV', contest_id='AvB_IRV',)
            assert assorter.assort(votes) == 1, f'{assorter.assort(votes)=}'

            votes = CVR.from_vote({'6': 1, '1': 2, '3': 3, '5': 4}, id='AvB_IRV', contest_id='AvB_IRV',)
            assert assorter.assort(votes) == 0, f'{assorter.assort(votes)=}'

            votes = CVR.from_vote({'3': 1, '47': 2}, id='AvB_IRV', contest_id='AvB_IRV',)
            assert assorter.assort(votes) == 0, f'{assorter.assort(votes)=}'

            votes = CVR.from_vote({})
            assert assorter.assort(votes) == 0.5, f'{assorter.assort(votes)=}'

            votes = CVR.from_vote({'6': 1, '47': 2}, id='AvB_IRV', contest_id='AvB_IRV',)
            assert assorter.assort(votes) == 0.5, f'{assorter.assort(votes)=}'

            votes = CVR.from_vote({'6': 1, '47': 2, '5': 3}, id='AvB_IRV', contest_id='AvB_IRV',)
            assert assorter.assort(votes) == 1, f'{assorter.assort(votes)=}'

            # winner-only assertion
            assorter = assertions['361']['28 v 50'].assorter

            votes = CVR.from_vote({'28': 1, '50': 2}, id='AvB_IRV', contest_id='AvB_IRV',)
            assert assorter.assort(votes) == 1, f'{assorter.assort(votes)=}'

            votes = CVR.from_vote({'28': 1}, id='AvB_IRV', contest_id='AvB_IRV',)
            assert assorter.assort(votes) == 1, f'{assorter.assort(votes)=}'

            votes = CVR.from_vote({'50': 1}, id='AvB_IRV', contest_id='AvB_IRV',)
            assert assorter.assort(votes) == 0, f'{assorter.assort(votes)=}'

            votes = CVR.from_vote({'27': 1, '28': 2}, id='AvB_IRV', contest_id='AvB_IRV',)
            assert assorter.assort(votes) == 0.5, f'{assorter.assort(votes)=}'

            votes = CVR.from_vote({'50': 1, '28': 2}, id='AvB_IRV', contest_id='AvB_IRV',)
            assert assorter.assort(votes) == 0, f'{assorter.assort(votes)=}'

            votes = CVR.from_vote({'27': 1, '26': 2}, id='AvB_IRV', contest_id='AvB_IRV',)
            assert assorter.assort(votes) == 0.5, f'{assorter.assort(votes)=}'

            votes = CVR.from_vote({})
            assert assorter.assort(votes) == 0.5, f'{assorter.assort(votes)=}'

            # elimination assertion
            assorter = assertions['361']['27 v 26 elim 28 50'].assorter

            votes = CVR.from_vote({'27': 1}, id='AvB_IRV', contest_id='AvB_IRV',)
            assert assorter.assort(votes) == 1, f'{assorter.assort(votes)=}'

            votes = CVR.from_vote({'50': 1, '27': 2}, id='AvB_IRV', contest_id='AvB_IRV',)
            assert assorter.assort(votes) == 1, f'{assorter.assort(votes)=}'

            votes = CVR.from_vote({'28': 1, '50': 2, '27': 3}, id='AvB_IRV', contest_id='AvB_IRV',)
            assert assorter.assort(votes) == 1, f'{assorter.assort(votes)=}'

            votes = CVR.from_vote({'28': 1, '27': 2, '50': 3}, id='AvB_IRV', contest_id='AvB_IRV',)
            assert assorter.assort(votes) == 1, f'{assorter.assort(votes)=}'

            votes = CVR.from_vote({'26': 1}, id='AvB_IRV', contest_id='AvB_IRV',)
            assert assorter.assort(votes) == 0, f'{assorter.assort(votes)=}'

            votes = CVR.from_vote({'50': 1, '26': 2}, id='AvB_IRV', contest_id='AvB_IRV',)
            assert assorter.assort(votes) == 0, f'{assorter.assort(votes)=}'

            votes = CVR.from_vote({'28': 1, '50': 2, '26': 3}, id='AvB_IRV', contest_id='AvB_IRV',)
            assert assorter.assort(votes) == 0, f'{assorter.assort(votes)=}'

            votes = CVR.from_vote({'28': 1, '26': 2, '50': 3}, id='AvB_IRV', contest_id='AvB_IRV',)
            assert assorter.assort(votes) == 0, f'{assorter.assort(votes)=}'

            votes = CVR.from_vote({'50': 1}, id='AvB_IRV', contest_id='AvB_IRV',)
            assert assorter.assort(votes) == 0.5, f'{assorter.assort(votes)=}'

            votes = CVR.from_vote({})
            assert assorter.assort(votes) == 0.5, f'{assorter.assort(votes)=}'

            votes = CVR.from_vote({'50': 1, '28': 2}, id='AvB_IRV', contest_id='AvB_IRV',)
            assert assorter.assort(votes) == 0.5, f'{assorter.assort(votes)=}'

            votes = CVR.from_vote({'28': 1, '50': 2}, id='AvB_IRV', contest_id='AvB_IRV',)
            assert assorter.assort(votes) == 0.5, f'{assorter.assort(votes)=}'

    def test_set_tally_pool_means(self, raw_AvB_asrtn):
        cvr_dicts = [{'id': 1, 'tally_pool': '1', 'pool': True, 'votes': {'AvB': {'Alice': 1}, 'CvD': {'Candy':True}}},
                     {'id': 2, 'tally_pool': '1', 'pool': True, 'votes': {'CvD': {'Elvis':True, 'Candy':False}, 'EvF': {}}},
                     {'id': 3, 'tally_pool': '1', 'pool': True, 'votes': {'GvH': {}}},
                     {'id': 4, 'tally_pool': '2', 'pool': True, 'votes': {'AvB': {'Bob': 1}, 'CvD': {'Candy':True}}},
                     {'id': 5, 'tally_pool': '2', 'pool': True, 'votes': {'CvD': {'Elvis':True, 'Candy':False}, 'EvF': {}}}
                   ]
        cvr_list = CVR.from_dict(cvr_dicts)
        tally_pools = CVR.pool_contests(cvr_list)  
        assert CVR.add_pool_contests(cvr_list, tally_pools) 
        #
        # without use_style
        raw_AvB_asrtn.assorter.set_tally_pool_means(cvr_list=cvr_list, tally_pools=tally_pools, use_style=False)
        np.testing.assert_almost_equal(raw_AvB_asrtn.assorter.tally_pool_means['1'], (1+1/2+1/2)/3)
        np.testing.assert_almost_equal(raw_AvB_asrtn.assorter.tally_pool_means['2'], (0+1/2)/2)
        #
        # with use_style, but contests have already been added to every CVR in each pool
        raw_AvB_asrtn.assorter.set_tally_pool_means(cvr_list=cvr_list, tally_pools=tally_pools, use_style=True)
        np.testing.assert_almost_equal(raw_AvB_asrtn.assorter.tally_pool_means['1'], (1+1/2+1/2)/3)
        np.testing.assert_almost_equal(raw_AvB_asrtn.assorter.tally_pool_means['2'], (0+1/2)/2)
        #
        # with use_style, without adding contests to every CVR in each pool
        cvr_dicts = [{'id': 1, 'tally_pool': '1', 'pool': True, 'votes': {'AvB': {'Alice': 1}, 'CvD': {'Candy':True}}},
                     {'id': 2, 'tally_pool': '1', 'pool': True, 'votes': {'CvD': {'Elvis':True, 'Candy':False}, 'EvF': {}}},
                     {'id': 3, 'tally_pool': '1', 'pool': True, 'votes': {'GvH': {}}},
                     {'id': 4, 'tally_pool': '2', 'pool': True, 'votes': {'AvB': {'Bob': 1}, 'CvD': {'Candy':True}}},
                     {'id': 5, 'tally_pool': '2', 'pool': True, 'votes': {'CvD': {'Elvis':True, 'Candy':False}, 'EvF': {}}}
                   ]
        cvr_list = CVR.from_dict(cvr_dicts)
        raw_AvB_asrtn.assorter.set_tally_pool_means(cvr_list=cvr_list, tally_pools=tally_pools, use_style=True)
        np.testing.assert_almost_equal(raw_AvB_asrtn.assorter.tally_pool_means['1'], 1)
        np.testing.assert_almost_equal(raw_AvB_asrtn.assorter.tally_pool_means['2'], 0)
        

    def test_count_tally_pool_vals(self, raw_AvB_asrtn):
        cvr_dicts = [{'id': 1, 'tally_pool': '1', 'pool': True, 'votes': {'AvB': {'Alice': 1}, 'CvD': {'Candy':True}}},
                     {'id': 2, 'tally_pool': '1', 'pool': True, 'votes': {'CvD': {'Elvis':True, 'Candy':False}, 'EvF': {}}},
                     {'id': 3, 'tally_pool': '1', 'pool': True, 'votes': {'GvH': {}}},
                     {'id': 4, 'tally_pool': '2', 'pool': True, 'votes': {'AvB': {'Bob': 1}, 'CvD': {'Candy':True}}},
                     {'id': 5, 'tally_pool': '2', 'pool': True, 'votes': {'CvD': {'Elvis':True, 'Candy':False}, 'EvF': {}}}
                   ]
        ref = {'1': defaultdict(int, {'n': 3, float(1): 1, float(0.5): 2}), 
                '2': defaultdict(int, {'n': 2, float(0): 1, float(1/2): 1})}
        cvr_list = CVR.from_dict(cvr_dicts)
        tally_pools = CVR.pool_contests(cvr_list)  
        assert CVR.add_pool_contests(cvr_list, tally_pools) 
        #
        # without use_style
        count_vals = raw_AvB_asrtn.assorter.count_tally_pool_vals(cvr_list=cvr_list, tally_pools=tally_pools, use_style=False)
        assert count_vals == ref
        count_vals = raw_AvB_asrtn.assorter.count_tally_pool_vals(cvr_list=cvr_list, tally_pools=tally_pools, use_style=False)
        assert count_vals == ref
        # with use_style, without adding contests to every CVR in each pool
        cvr_dicts = [{'id': 1, 'tally_pool': '1', 'pool': True, 'votes': {'AvB': {'Alice': 1}, 'CvD': {'Candy':True}}},
                     {'id': 2, 'tally_pool': '1', 'pool': True, 'votes': {'CvD': {'Elvis':True, 'Candy':False}, 'EvF': {}}},
                     {'id': 3, 'tally_pool': '1', 'pool': True, 'votes': {'GvH': {}}},
                     {'id': 4, 'tally_pool': '2', 'pool': True, 'votes': {'AvB': {'Bob': 1}, 'CvD': {'Candy':True}}},
                     {'id': 5, 'tally_pool': '2', 'pool': True, 'votes': {'CvD': {'Elvis':True, 'Candy':False}, 'EvF': {}}}
                   ]
        cvr_list = CVR.from_dict(cvr_dicts)
        count_vals = raw_AvB_asrtn.assorter.count_tally_pool_vals(cvr_list=cvr_list, tally_pools=tally_pools, use_style=True)
        ref = {'1': defaultdict(int, {'n': 1, float(1): 1}), '2': defaultdict(int, {'n': 1, float(0): 1})}
        assert count_vals == ref

    def test_overstatement(self, con_test):
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

        aVb = Assertion(contest=con_test, assorter=Assorter(contest=con_test,
                        assort = (lambda c, contest_id="AvB", winr="Alice", losr="Bob":
                        ( CVR.as_vote(c.get_vote_for("AvB", winr))
                        - CVR.as_vote(c.get_vote_for("AvB", losr))
                        + 1)/2), upper_bound=1))
        assert aVb.assorter.overstatement(mvrs[0], cvrs[0], use_style=True) == 0
        assert aVb.assorter.overstatement(mvrs[0], cvrs[0], use_style=False) == 0

        assert aVb.assorter.overstatement(mvrs[0], cvrs[1], use_style=True) == -1
        assert aVb.assorter.overstatement(mvrs[0], cvrs[1], use_style=False) == -1

        assert aVb.assorter.overstatement(mvrs[2], cvrs[0], use_style=True) == 1/2
        assert aVb.assorter.overstatement(mvrs[2], cvrs[0], use_style=False) == 1/2

        assert aVb.assorter.overstatement(mvrs[2], cvrs[1], use_style=True) == -1/2
        assert aVb.assorter.overstatement(mvrs[2], cvrs[1], use_style=False) == -1/2


        assert aVb.assorter.overstatement(mvrs[1], cvrs[0], use_style=True) == 1
        assert aVb.assorter.overstatement(mvrs[1], cvrs[0], use_style=False) == 1

        assert aVb.assorter.overstatement(mvrs[2], cvrs[0], use_style=True) == 1/2
        assert aVb.assorter.overstatement(mvrs[2], cvrs[0], use_style=False) == 1/2

        assert aVb.assorter.overstatement(mvrs[3], cvrs[0], use_style=True) == 1
        assert aVb.assorter.overstatement(mvrs[3], cvrs[0], use_style=False) == 1/2

        try:
            tst = aVb.assorter.overstatement(mvrs[3], cvrs[3], use_style=True)
            raise AssertionError('aVb is not contained in the mvr or cvr')
        except ValueError:
            pass
        assert aVb.assorter.overstatement(mvrs[3], cvrs[3], use_style=False) == 0

        assert aVb.assorter.overstatement(mvrs[4], cvrs[4], use_style=True) == 1/2
        assert aVb.assorter.overstatement(mvrs[4], cvrs[4], use_style=False) == 1/2
        assert aVb.assorter.overstatement(mvrs[4], cvrs[4], use_style=False) == 1/2
        assert aVb.assorter.overstatement(mvrs[4], cvrs[0], use_style=True) == 1
        assert aVb.assorter.overstatement(mvrs[4], cvrs[0], use_style=False) == 1
        assert aVb.assorter.overstatement(mvrs[4], cvrs[1], use_style=True) == 0
        assert aVb.assorter.overstatement(mvrs[4], cvrs[1], use_style=False) == 0


    def test_overstatement_assorter(self, con_test):
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


        aVb = Assertion(contest=con_test, assorter=Assorter(contest=con_test,
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


    def test_assorter_sample_size(self, AvB_plur):
        # Test Assorter.sample_size using the Kaplan-Wald risk function
        N = int(10**4)
        loser = list(set(AvB_plur.candidates)-set(AvB_plur.winner))
        AvB_plur.assertions = Assertion.make_plurality_assertions(AvB_plur, winner=AvB_plur.winner, loser=loser)
        AvB_plur.find_margins_from_tally()
        for a_id, a in AvB_plur.assertions.items():
            # first test
            rate_1=0.01
            rate_2=0.001
            sam_size1 = a.find_sample_size(data=np.ones(10), prefix=True, rate_1=rate_1, reps=None, quantile=0.5, 
                                           seed=1234567890)
            # Kaplan-Markov martingale is \prod (t+g)/(x+g). For x = [1, 1, ...], sample size should be:
            ss1 = math.ceil(np.log(AvB_plur.risk_limit)/np.log((a.test.t+a.test.g)/(1+a.test.g)))
            assert sam_size1 == ss1
            #
            # second test
            # For "clean", the term is (1/2+g)/(clean+g); for a one-vote overstatement, it is (1/2+g)/(one_over+g).
            sam_size2 = a.find_sample_size(data=None, prefix=True, rate_1=rate_1, reps=10**2, quantile=0.5, 
                                           seed=1234567890)
            clean = 1/(2-a.margin/a.assorter.upper_bound)
            over = clean/2 # corresponds to an overstatement of upper_bound/2, i.e., 1 vote.
            c = (a.test.t+a.test.g)/(clean+a.test.g)
            o = (a.test.t+a.test.g)/(clean/2+a.test.g)
            # the following calculation assumes the audit will terminate before the second overstatement error
            ss2 = math.ceil(np.log(AvB_plur.risk_limit/o)/np.log(c))+1
            assert sam_size2 == ss2
            #
            # third test
            sam_size3 = a.find_sample_size(data=None, prefix=True, rate_1=rate_1, rate_2=rate_2,
                                           reps=10**2, quantile=0.99, seed=1234567890)
            assert sam_size3 > sam_size2

    def test_find_margin_from_tally(self):
        AvB = Contest.from_dict({'id': 'AvB',
                     'name': 'AvB',
                     'risk_limit': 0.01,
                     'cards': 10**4,
                     'choice_function': Contest.SOCIAL_CHOICE_FUNCTION.PLURALITY,
                     'n_winners': 1,
                     'candidates': ['Alice','Bob','Carol'],
                     'winner': ['Alice'],
                     'audit_type': Audit.AUDIT_TYPE.CARD_COMPARISON,
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

    def test_interleave_values(self):
        n_small = 5
        n_med = 3
        n_big = 6
        x = Assertion.interleave_values(n_small, n_med, n_big)
        assert len(x) == 14
        assert x[0] == 0
        assert np.sum(x==0) == 5
        assert np.sum(x==1/2) == 3
        assert np.sum(x==1) == 6
        
        n_small = 0
        n_med = 3
        n_big = 6
        big = 2
        med = 1
        small = 0.1
        x = Assertion.interleave_values(n_small, n_med, n_big, small=small, med=med, big=big)
        assert len(x) == 9
        assert x[0] == 1
        assert np.sum(x==0.1) == 0
        assert np.sum(x==1) == 3
        assert np.sum(x==2) == 6


##########################################################################################
if __name__ == "__main__":
    sys.exit(pytest.main(["-qq"], plugins=None))
