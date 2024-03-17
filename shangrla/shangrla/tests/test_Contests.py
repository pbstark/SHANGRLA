import math
import numpy as np
import scipy as sp
import pandas as pd
import json
import csv
import warnings
import typing
import sys
import pytest
import coverage
from numpy import testing
from collections import OrderedDict, defaultdict
from cryptorandom.cryptorandom import SHA256, random, int_from_hash
from cryptorandom.sample import random_permutation
from cryptorandom.sample import sample_by_index


from shangrla.Audit import Audit, Assertion, Assorter, Contest, CVR, Stratum
from shangrla.NonnegMean import NonnegMean
from shangrla.Dominion import Dominion
from shangrla.Hart import Hart

###################################################################################################
class TestContests:

    contest_dict = {
                 'AvB': {
                 'name': 'contest_1',
                 'risk_limit': 0.05,
                 'cards': 10**4,
                 'choice_function': Contest.SOCIAL_CHOICE_FUNCTION.PLURALITY,
                 'n_winners': 2,
                 'candidates': 5,
                 'candidates': ['alice','bob','carol','dave','erin'],
                 'winner': ['alice','bob'],
                 'audit_type': Audit.AUDIT_TYPE.CARD_COMPARISON,
                 'test': NonnegMean.alpha_mart,
                 'use_style': True
                },
                 'CvD': {
                 'name': 'contest_2',
                 'risk_limit': 0.05,
                 'cards': 10**4,
                 'choice_function': Contest.SOCIAL_CHOICE_FUNCTION.SUPERMAJORITY,
                 'n_winners': 1,
                 'candidates': 4,
                 'candidates': ['alice','bob','carol','dave'],
                 'winner': ['alice'],
                 'audit_type': Audit.AUDIT_TYPE.POLLING,
                 'test': NonnegMean.alpha_mart,
                 'use_style': False
                },
                 'EvF': {
                 'name': 'contest_3',
                 'risk_limit': 0.05,
                 'cards': 10**4,
                 'choice_function': Contest.SOCIAL_CHOICE_FUNCTION.IRV,
                 'n_winners': 1,
                 'candidates': 4,
                 'candidates': ['alice','bob','carol','dave'],
                 'winner': ['alice'],
                 'audit_type': Audit.AUDIT_TYPE.CARD_COMPARISON,
                 'test': NonnegMean.alpha_mart,
                 'use_style': False
                }
            }

    def test_contests_from_dict_of_dicts(self):
        ids = ['1','2']
        atts = ('id','name','risk_limit','cards','choice_function','n_winners','share_to_win','candidates',
                'winner','assertion_file','audit_type','test','use_style')
        contests = Contest.from_dict_of_dicts(self.contest_dict)
        for id, c in contests.items():
            assert c.__dict__.get('id') == id
            for att in atts:
                if att != 'id':
                    assert c.__dict__.get(att) == self.contest_dict[id].get(att)

    def test_tally(self):
        cvr_dict = [{'id': 1, 'votes': {'AvB': {'Alice':True}, 'CvD': {'Candy':True}}},
                    {'id': 2, 'votes': {'AvB': {'Bob':True}, 'CvD': {'Elvis':True, 'Candy':False}}},
                    {'id': 3, 'votes': {'EvF': {'Bob':1, 'Edie':2}, 'CvD': {'Elvis':False, 'Candy':True}}},
                    {'id': 4, 'votes': {'AvB': {'Alice':1}, 'CvD': {'Candy':'yes'}}},
                    {'id': 5, 'votes': {'AvB': {'Bob':True}, 'CvD': {'Elvis':True, 'Candy':False}}},
                    {'id': 6, 'votes': {'EvF': {'Bob':2, 'Edie':1}, 'CvD': {'Elvis':False, 'Candy':True}}},
                    {'id': 7, 'votes': {'AvB': {'Alice':2}, 'CvD': {'Elvis':False, 'Candy':True}}}
                   ]
        cvr_list = CVR.from_dict(cvr_dict)
        contests = Contest.from_dict_of_dicts(self.contest_dict)
        Contest.tally(contests, cvr_list)
        assert contests['AvB'].tally == {'Alice': 3, 'Bob': 3}
        assert contests['CvD'].tally == {'Candy': 5, 'Elvis': 2}
        assert contests['EvF'].tally is None 
        # TO DO: assert that this raises a warning about contest EvF
            

##########################################################################################
if __name__ == "__main__":
    sys.exit(pytest.main(["-qq"], plugins=None))
