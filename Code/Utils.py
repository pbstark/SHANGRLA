import math
import numpy as np
import scipy as sp
from scipy.stats import bernoulli
import pandas as pd
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
from Audit import Audit, Assertion, Assorter, Contest, Stratum
from NonnegMean import NonnegMean


class Utils:
    '''
    Utilities for SHANGRLA RLAs
    '''
    
    @classmethod
    def check_audit_parameters(cls, audit: object=None, contests: dict=None):
        '''
        Check whether the audit parameters are valid; complain if not.

        Parameters:
        ---------
        audit: Audit
            general information about the audit
        
        contests: dict of Contests
            contest-specific information for the audit

        Returns:
        --------
        '''
        assert error_rate >= 0, 'expected error rate must be nonnegative'
        for i, c in contests.items():
            assert c.risk_limit > 0, 'risk limit must be nonnegative in ' + c + ' contest'
            assert c.risk_limit < 1, 'risk limit must be less than 1 in ' + c + ' contest'
            assert c.choice_function in Audit.SOCIAL_CHOICE_FUNCTION.SOCIAL_CHOICE_FUNCTIONS, \
                      f'unsupported choice function {c.choice_function} in contest {i}'
            assert c.n_winners <= len(c.candidates), f'more winners than candidates in contest {i}'
            assert len(c.reported_winners) == c.n_winners, \
                f'number of reported winners does not equal n_winners in contest {i}'
            for w in c.reported_winners:
                assert w in c.candidates, f'reported winner {w} is not a candidate in contest {i}'
            if c.choice_function in [Audit.SOCIAL_CHOICE_FUNCTION.IRV, Audit.SOCIAL_CHOICE_FUNCTION.SUPERMAJORITY]:
                assert c.n_winners == 1, f'{c.choice_function} can have only 1 winner in contest {i}'
            if c.choice_function == Audit.SOCIAL_CHOICE_FUNCTION.IRV:
                assert c.assertion_file, f'IRV contest {i} requires an assertion file'


    @classmethod
    def summarize_status(cls, audit: object=None, contests: dict=None):
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
        for c in contests:
            print(f'p-values for assertions in contest {c}')
            cpmax = 0
            for a in contests[c],assertions:
                cpmax = np.max([cpmax,contests[c].assertions[a].p_value])
                print(a, contests[c].assertions[a].p_value)
            if cpmax <= contests[c].risk_limit:
                print(f'\ncontest {c} AUDIT COMPLETE at risk limit {contests[c].risk_limit}. Attained risk {cpmax}')
            else:
                done = False
                print(f'\ncontest {c} audit INCOMPLETE at risk limit {contest[c].risk_limit}. Attained risk {cpmax}')
                print("assertions remaining to be proved:")
                for a in contests[c].assertions:
                    if contests[c].assertions[a].p_value > contests[c].risk_limit:
                        print(f'{a}: current risk {contests[c].assertions[a].p_value}')
        return done

    @classmethod
    def write_audit_parameters(cls, audit: object=None, contests: dict=None):
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
        log_file = audit.log_file
        out = {'Audit': audit,
               'contests': contests
              }
        with open(log_file, 'w') as f:
            f.write(json.dumps(out, cls=NpEncoder))

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

class NpEncoder(json.JSONEncoder):
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
