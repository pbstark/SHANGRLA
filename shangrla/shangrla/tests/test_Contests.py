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

    def test_contests_from_dict_of_dicts(self):
        ids = ['1','2']
        choice_functions = [Contest.SOCIAL_CHOICE_FUNCTION.PLURALITY, Contest.SOCIAL_CHOICE_FUNCTION.SUPERMAJORITY]
        risk_limit = 0.05
        cards = [10000, 20000]
        n_winners = [2, 1]
        candidates = [4, 2]
        audit_type = [Audit.AUDIT_TYPE.POLLING, Audit.AUDIT_TYPE.CARD_COMPARISON]
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
                 'audit_type': Audit.AUDIT_TYPE.CARD_COMPARISON,
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
if __name__ == "__main__":
    sys.exit(pytest.main(["-qq"], plugins=None))
