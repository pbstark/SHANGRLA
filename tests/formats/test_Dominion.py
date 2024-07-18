import pandas as pd
import sys
import pytest

from shangrla.formats.Dominion import Dominion

##########################################################################################

class TestDominion:
    def test_read_cvrs_old_format(self):
        """
        Test reading the older Dominion CVR format
        """
        cvr_list = Dominion.read_cvrs(
            "tests/core/data/Dominion_CVRs/test_5.2.18.2.Dominion.json", pool_groups = [1]
        )
        assert len(cvr_list) == 2
        cvr_1, cvr_2 = cvr_list
        assert cvr_1.id == "60001_1_1"
        assert cvr_1.tally_pool == "60001_1"
        assert cvr_1.pool, f'{cvr_1.pool=}'
        assert list(cvr_1.votes.keys()) == ["111"]
        assert cvr_1.votes["111"] == {"6": 1, "1": 2}
        assert cvr_1.get_vote_for("111", "6")
        assert cvr_1.get_vote_for("111", "1")
        assert cvr_1.get_vote_for("111", "999") is False
        assert cvr_2.id == "60009_3_21"
        assert cvr_2.tally_pool == "60009_3"
        assert not cvr_2.pool, f'{cvr_2.pool=}'
        assert list(cvr_2.votes.keys()) == ["111", "122"]
        assert cvr_2.votes["111"] == {"6": 1}
        assert cvr_2.votes["122"] == {"9": 1, "48": 2}
        assert cvr_2.get_vote_for("111", "6")
        assert cvr_2.get_vote_for("111", "1") is False
        assert cvr_2.get_vote_for("122", "9")
        assert cvr_2.get_vote_for("122", "48")
        assert cvr_2.get_vote_for("122", "999") is False

    def test_read_cvrs_old_format_adjudicated(self):
        """
        Tests using the "Modified" (adjudicated) data from the CVR (if present) over the "Original"
        Ingests the older Dominion CVR format
        """
        cvr_list = Dominion.read_cvrs(
            "tests/core/data/Dominion_CVRs/test_5.2.18.2.Dominion.json",
            use_adjudicated=True,
        )
        assert len(cvr_list) == 2
        cvr_1, cvr_2 = cvr_list
        assert cvr_1.id == "60001_1_1"
        assert cvr_1.tally_pool == "60001_1"
        assert not cvr_1.pool, f'{cvr_1.pool=}'
        assert list(cvr_1.votes.keys()) == ["111"]
        assert cvr_1.votes["111"] == {"6": 1, "1": 2}
        assert cvr_1.get_vote_for("111", "6")
        assert cvr_1.get_vote_for("111", "1")
        assert cvr_1.get_vote_for("111", "999") is False
        assert cvr_2.id == "60009_3_21"
        assert cvr_2.tally_pool == "60009_3"
        assert not cvr_2.pool, f'{cvr_2.pool=}'
        assert list(cvr_2.votes.keys()) == ["111", "122"]
        assert cvr_2.votes["111"] == {"6": 1}
        assert cvr_2.votes["122"] == {"9": 1}
        assert cvr_2.get_vote_for("111", "6")
        assert cvr_2.get_vote_for("111", "1") is False
        assert cvr_2.get_vote_for("122", "9")
        assert cvr_2.get_vote_for("122", "48") is False
        assert cvr_2.get_vote_for("122", "999") is False

    def test_read_cvrs_new_format(self):
        """
        Test reading the newer Dominion CVR format
        """
        cvr_list = Dominion.read_cvrs(
            "tests/core/data/Dominion_CVRs/test_5.10.50.85.Dominion.json"
        )
        assert len(cvr_list) == 2
        cvr_1, cvr_2 = cvr_list
        assert cvr_1.id == "1_2_13"
        assert cvr_1.tally_pool == "1_2"
        assert list(cvr_1.votes.keys()) == ["1"]
        assert cvr_1.votes["1"] == {"5": 1}
        assert cvr_1.get_vote_for("1", "5")
        assert cvr_1.get_vote_for("1", "999") is False
        assert cvr_2.id == "1_5_119"
        assert cvr_2.tally_pool == "1_5"
        assert list(cvr_2.votes.keys()) == ["1"]
        assert cvr_2.votes["1"] == {"6": 1}
        assert cvr_2.get_vote_for("1", "6")
        assert cvr_2.get_vote_for("1", "999") is False

    def test_read_cvrs_new_format_adjudicated(self):
        """
        Tests using the "Modified" (adjudicated) data from the CVR (if present) over the "Original"
        Ingests the newer Dominion CVR format
        """
        cvr_list = Dominion.read_cvrs(
            "tests/core/data/Dominion_CVRs/test_5.10.50.85.Dominion.json",
            use_adjudicated=True
        )
        assert len(cvr_list) == 2
        cvr_1, cvr_2 = cvr_list
        assert cvr_1.id == "1_2_13"
        assert cvr_1.tally_pool == "1_2"
        assert not cvr_1.pool
        assert list(cvr_1.votes.keys()) == ["1"]
        assert cvr_1.votes["1"] == {"5": 1}
        assert cvr_1.get_vote_for("1", "5")
        assert cvr_1.get_vote_for("1", "999") is False
        assert cvr_2.id == "1_5_119"
        assert cvr_2.tally_pool == "1_5"
        assert not cvr_2.pool
        assert list(cvr_2.votes.keys()) == ["1"]
        assert cvr_2.votes["1"] == {}
        assert cvr_2.get_vote_for("1", "6") is False
        assert cvr_2.get_vote_for("1", "999") is False

    def test_read_cvrs_new_format_vbm_only(self):
        """
        Test reading the newer Dominion CVR format, selecting VBM results only
        """
        cvr_list = Dominion.read_cvrs(
            "tests/core/data/Dominion_CVRs/test_5.10.50.85.Dominion.json",
            include_groups=[2]
        )
        assert len(cvr_list) == 1
        cvr_2 = cvr_list[0]
        assert cvr_2.id == "1_5_119"
        assert cvr_2.tally_pool == "1_5"
        assert list(cvr_2.votes.keys()) == ["1"]
        assert cvr_2.votes["1"] == {"6": 1}
        assert cvr_2.get_vote_for("1", "6")
        assert cvr_2.get_vote_for("1", "999") is False

    def test_read_cvrs_directory(self):
        """
        Tests the convenience function for reading all CVRs from a given directory
        """
        cvr_list = Dominion.read_cvrs_directory(
            "tests/core/data/Dominion_CVRs/CVR_Export", pool_groups=[2]
        )
        assert len(cvr_list) == 2
        cvr_1, cvr_2 = cvr_list
        assert cvr_1.id == "1_2_13"
        assert cvr_1.tally_pool == "1_2"
        assert not cvr_1.pool, f'{cvr_1.pool=}'
        assert list(cvr_1.votes.keys()) == ["1"]
        assert cvr_1.votes["1"] == {"5": 1}
        assert cvr_1.get_vote_for("1", "5")
        assert cvr_1.get_vote_for("1", "999") is False
        assert cvr_2.id == "1_5_119"
        assert cvr_2.tally_pool == "1_5"
        assert cvr_2.pool, f'{cvr_2.pool=}'
        assert list(cvr_2.votes.keys()) == ["1"]
        assert cvr_2.votes["1"] == {"6": 1}
        assert cvr_2.get_vote_for("1", "6")
        assert cvr_2.get_vote_for("1", "999") is False

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
