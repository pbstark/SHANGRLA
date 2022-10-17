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
# from Audit import Audit, Assertion, Assorter, Contest
from NonnegMean import NonnegMean


class Utils:
    '''
    Utilities for SHANGRLA RLAs
    '''
    
    @classmethod
    def check_audit_parameters(cls, risk_function, error_rate, contests):
        '''
        Check whether the audit parameters are valid; complain if not.

        Parameters:
        ---------
        risk_function: string
            the name of the risk-measuring function for the audit

        error_rate: float
            expected rate of 1-vote overstatements

        contests: dict of dicts
            contest-specific information for the audit

        Returns:
        --------
        '''
        assert error_rate >= 0, 'expected error rate must be nonnegative'
        for c in contests.keys():
            assert contests[c]['risk_limit'] > 0, 'risk limit must be nonnegative in ' + c + ' contest'
            assert contests[c]['risk_limit'] < 1, 'risk limit must be less than 1 in ' + c + ' contest'
            assert contests[c]['choice_function'] in ['IRV','plurality','supermajority'], \
                      'unsupported choice function ' + contests[c]['choice_function'] + ' in ' \
                      + c + ' contest'
            assert contests[c]['n_winners'] <= len(contests[c]['candidates']), \
                'fewer candidates than winners in ' + c + ' contest'
            assert len(contests[c]['reported_winners']) == contests[c]['n_winners'], \
                'number of reported winners does not equal n_winners in ' + c + ' contest'
            for w in contests[c]['reported_winners']:
                assert w in contests[c]['candidates'], \
                    'reported winner ' + w + ' is not a candidate in ' + c + 'contest'
            if contests[c]['choice_function'] in ['IRV','supermajority']:
                assert contests[c]['n_winners'] == 1, \
                    contests[c]['choice_function'] + ' can have only 1 winner in ' + c + ' contest'
            if contests[c]['choice_function'] == 'IRV':
                assert contests[c]['assertion_file'], 'IRV contest ' + c + ' requires an assertion file'
            if contests[c]['choice_function'] == 'supermajority':
                assert contests[c]['share_to_win'] >= 0.5, \
                    'super-majority contest requires winning at least 50% of votes in ' + c + ' contest'


    @classmethod
    def summarize_status(cls, contests):
        '''
        Determine whether the audit of individual assertions, contests, and the whole election are finished.

        Prints a summary.

        Parameters:
        -----------
        contests: dict of dicts
            dict of contest information

        Returns:
        --------
        done: boolean
            is the audit finished?'''
        done = True
        for c in contests:
            print("p-values for assertions in contest {}".format(c))
            cpmax = 0
            for a in contests[c]['assertions']:
                cpmax = np.max([cpmax,contests[c]['assertions'][a].p_value])
                print(a, contests[c]['assertions'][a].p_value)
            if cpmax <= contests[c]['risk_limit']:
                print("\ncontest {} AUDIT COMPLETE at risk limit {}. Attained risk {}".format(\
                    c, contests[c]['risk_limit'], cpmax))
            else:
                done = False
                print("\ncontest {} audit INCOMPLETE at risk limit {}. Attained risk {}".format(\
                    c, contests[c]['risk_limit'], cpmax))
                print("assertions remaining to be proved:")
                for a in contests[c]['assertions']:
                    if contests[c]['assertions'][a].p_value > contests[c]['risk_limit']:
                        print("{}: current risk {}".format(a, contests[c]['assertions'][a].p_value))
        return done

    @classmethod
    def write_audit_parameters(
                               cls, log_file, seed, replacement, risk_function,
                               max_cards, n_cvrs, manifest_cards, phantom_cards, error_rate, contests):
        '''
        Write audit parameters to log_file as a json structure

        Parameters:
        ---------
        log_file: string
            filename to write to

        seed: string
            seed for the PRNG for sampling ballots

        risk_function: string
            name of the risk-measuring function used in the audit

        error_rate: float
            expected rate of 1-vote overstatements

        contests: dict of dicts
            contest-specific information for the audit

        Returns:
        --------
        no return value
        '''
        out = {"seed": seed,
               "replacement": replacement,
               "risk_function": risk_function,
               "max_cards": int(max_cards),
               "n_cvrs": int(n_cvrs),
               "manifest_cards": int(manifest_cards),
               "phantom_cards": int(phantom_cards),
               "error_rate": error_rate,
               "contests": contests
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
        return super(NpEncoder, self).default(obj)

################
# Unit tests


if __name__ == "__main__":
    print('No unit tests implemented')
