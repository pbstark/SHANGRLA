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
        
##########################################################################################
if __name__ == "__main__":
    sys.exit(pytest.main(["-qq"], plugins=None))
