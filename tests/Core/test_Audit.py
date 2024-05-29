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


from shangrla.core.Audit import Audit, Assertion, Assorter, Contest, CVR, Stratum
from shangrla.core.NonnegMean import NonnegMean
from shangrla.core.Dominion import Dominion
from shangrla.core.Hart import Hart

#######################################################################################################

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
                                          'audit_type':  Audit.AUDIT_TYPE.CARD_COMPARISON,
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

    
##########################################################################################
if __name__ == "__main__":
    sys.exit(pytest.main(["-qq"], plugins=None))
