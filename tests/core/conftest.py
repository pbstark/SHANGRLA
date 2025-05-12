# fixtures to configure unit tests

import math
import numpy as np
import sys
import pytest

from shangrla.core.Audit import Audit, Assertion, Assorter, Contest, CVR
from shangrla.core.NonnegMean import NonnegMean

@pytest.fixture
def con_test():
    return Contest.from_dict({'id': 'AvB',
         'name': 'AvB',
         'risk_limit': 0.05,
         'cards': 10**4,
         'choice_function': Contest.SOCIAL_CHOICE_FUNCTION.SUPERMAJORITY,
         'n_winners': 1,
         'candidates': 3,
         'candidates': ['Alice','Bob','Candy'],
         'winner': ['Alice'],
         'audit_type': Audit.AUDIT_TYPE.CARD_COMPARISON,
         'share_to_win': 2/3,
         'test': NonnegMean.alpha_mart,
         'use_style': True
        })

@pytest.fixture
def plur_cvr_list():
    return CVR.from_dict([
       {'id': "1_1", 'tally_pool': '1', 'votes': {'AvB': {'Alice':True}}},
       {'id': "1_2", 'tally_pool': '1', 'votes': {'AvB': {'Bob':True}}},
       {'id': "1_3", 'tally_pool': '2', 'votes': {'AvB': {'Alice':True}}},
       {'id': "1_4", 'tally_pool': '2', 'votes': {'AvB': {'Alice':True}}}])
    
@pytest.fixture
def plur_con_test():
    return Contest.from_dict({'id': 'AvB',
        'name': 'AvB',
        'risk_limit': 0.05,
        'cards': 4,
        'choice_function': Contest.SOCIAL_CHOICE_FUNCTION.PLURALITY,
        'n_winners': 1,
        'candidates': 3,
        'candidates': ['Alice','Bob','Candy'],
        'winner': ['Alice'],
        'audit_type': Audit.AUDIT_TYPE.CARD_COMPARISON,
        'test': NonnegMean.alpha_mart,
        'estim': NonnegMean.optimal_comparison,
        'use_style': True
        })

#assertion without a margin
@pytest.fixture
def raw_AvB_asrtn(plur_con_test):
    return Assertion(
        contest = plur_con_test,
        winner = "Alice",
        loser = "Bob",
        test = NonnegMean(test=plur_con_test.test, estim=plur_con_test.estim, u=1,
                            N=plur_con_test.cards, t=1/2, random_order=True),
        assorter = Assorter(
            contest = plur_con_test,
            assort = lambda c:
                         (CVR.as_vote(c.get_vote_for("AvB", "Alice"))
                         - CVR.as_vote(c.get_vote_for("AvB", "Bob"))
                          + 1)/2,
            upper_bound = 1
        )
    )

@pytest.fixture
def AvB_plur():
    return Contest.from_dict({'id': 'AvB_plur',
         'name': 'AvB_plur',
         'risk_limit': 0.05,
         'cards': 7000,
         'choice_function': Contest.SOCIAL_CHOICE_FUNCTION.PLURALITY,
         'n_winners': 1,
         'candidates': ['Alice','Bob', 'Carol'],
         'winner': ['Alice'],
         'audit_type': Audit.AUDIT_TYPE.CARD_COMPARISON,
         'test': NonnegMean.kaplan_markov,
         'tally': {'Alice': 3000, 'Bob': 2000, 'Carol': 1000},
         'g': 0.1,
         'use_style': True
          }
    )
        
@pytest.fixture
def AvB_IRV():
    return Contest.from_dict({'id': 'AvB_IRV',
         'name': 'AvB_IRV',
         'risk_limit': 0.05,
         'cards': 29776,
         'choice_function': Contest.SOCIAL_CHOICE_FUNCTION.IRV,
         'n_winners': 1,
         'test': NonnegMean.alpha_mart,
         'use_style': True
         }
    )

#comparison and polling audits referencing plur_cvr_list
@pytest.fixture
def comparison_audit():
    return Audit.from_dict(
        {'quantile':       0.8,
         'error_rate_1': 0,
         'error_rate_2': 0,
         'reps':           100,
         'strata':         {'stratum_1': {'max_cards':   4,
                                          'use_style':   True,
                                          'replacement': False,
                                          'audit_type':  Audit.AUDIT_TYPE.CARD_COMPARISON,
                                          'test':        NonnegMean.alpha_mart,
                                          'estimator':   NonnegMean.optimal_comparison,
                                          'test_kwargs': {}
                                         }
                           }
        })
