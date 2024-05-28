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



######################################################################################
class TestHart:
    def test_read_cvrs_directory(self):
        cvr_list = Hart.read_cvrs_directory("shangrla/shangrla/tests/Data/Hart_CVRs")
        cvr_1 = cvr_list[0]
        cvr_2 = cvr_list[1]
        assert list(cvr_1.votes.keys()) == ["PRESIDENT","GOVERNOR","MAYOR"]
        assert cvr_1.votes['GOVERNOR'] == {}
        assert cvr_2.get_vote_for("MAYOR", "WRITE_IN")
        assert cvr_2.get_vote_for("PRESIDENT", "George Washington")

    def test_read_cvrs_zip(self):
        cvr_list = Hart.read_cvrs_zip("shangrla/shangrla/tests/Data/Hart_CVRs.zip")
        cvr_1 = cvr_list[0]
        cvr_2 = cvr_list[1]
        assert list(cvr_1.votes.keys()) == ["PRESIDENT","GOVERNOR","MAYOR"]
        assert cvr_1.votes['GOVERNOR'] == {}
        assert cvr_2.get_vote_for("MAYOR", "WRITE_IN")
        assert cvr_2.get_vote_for("PRESIDENT", "George Washington")


    def test_prep_manifest(self):
        # without phantoms
        manifest_f = pd.read_excel("shangrla/shangrla/tests/Data/Hart_manifest.xlsx")
        max_cards = 1141765
        n_cvrs = 1141765
        manifest, manifest_cards, phantoms = Hart.prep_manifest(manifest_f, max_cards, n_cvrs)
        assert manifest['Number of Ballots'].astype(int).sum() == max_cards
        assert phantoms == 0
        # with phantoms
        manifest_f = pd.read_excel("shangrla/shangrla/tests/Data/Hart_manifest.xlsx")
        max_cards = 1500000
        manifest, manifest_cards, phantoms = Hart.prep_manifest(manifest_f, max_cards, n_cvrs)
        assert manifest['Number of Ballots'].astype(int).sum() == max_cards
        assert phantoms == max_cards - n_cvrs

    def test_sample_from_manifest(self):
        cvr_dict = [{'id': "1_1", 'votes': {'AvB': {'Alice':True}}},
                    {'id': "1_2", 'votes': {'AvB': {'Bob':True}}},
                    {'id': "1_3", 'votes': {'AvB': {'Alice':True}}}]
        manifest_f = pd.DataFrame.from_dict({'Container': ['Mail', 'Mail'], 'Tabulator': [1, 1],\
            'Batch Name': [1, 2], 'Number of Ballots': [1, 2]}, orient = "columns")
        manifest, manifest_cards, phantoms = Hart.prep_manifest(manifest_f, 3, 3)
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
