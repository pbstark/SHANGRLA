import pandas as pd
import sys
import pytest
from pathlib import Path

from shangrla.formats.Dominion import Dominion
from shangrla.core.Audit import CVR

##########################################################################################


class TestDominion:
    def test_read_cvrs_old_format_rules_not_enforced(self):
        """
        Test reading the older Dominion CVR format without rules enforced
        """
        cvr_list = Dominion.read_cvrs(
            "tests/core/data/Dominion_CVRs/test_5.2.18.2.Dominion.json",
            use_current=False,
            enforce_rules=False,
            pool_groups=[1],
        )
        assert len(cvr_list) == 2
        cvr_1, cvr_2 = cvr_list
        assert cvr_1.id == "60001-1-1"
        assert cvr_1.tally_pool == "60001-1"
        assert cvr_1.pool, f"{cvr_1.pool=}"
        assert list(cvr_1.votes.keys()) == ["111"]
        assert cvr_1.votes["111"] == {"6": 1, "1": 2}
        assert cvr_1.get_vote_for("111", "6")
        assert cvr_1.get_vote_for("111", "1")
        assert cvr_1.get_vote_for("111", "999") is False
        assert cvr_2.id == "60009-3-21"
        assert cvr_2.tally_pool == "60009-3"
        assert not cvr_2.pool, f"{cvr_2.pool=}"
        assert list(cvr_2.votes.keys()) == ["111", "122"]
        assert cvr_2.votes["111"] == {"6": 1}
        assert cvr_2.votes["122"] == {"9": 1, "48": 2}
        assert cvr_2.get_vote_for("111", "6")
        assert cvr_2.get_vote_for("111", "1") is False
        assert cvr_2.get_vote_for("122", "9")
        assert cvr_2.get_vote_for("122", "48")
        assert cvr_2.get_vote_for("122", "999") is False

    def test_read_cvrs_old_format_rules_enforced(self):
        """
        Tests using the "Modified" (adjudicated) data from the CVR (if present) over the "Original"
        Ingests the older Dominion CVR format
        """
        cvr_list = Dominion.read_cvrs(
            "tests/core/data/Dominion_CVRs/test_5.2.18.2.Dominion.json",
        )
        assert len(cvr_list) == 2
        cvr_1, cvr_2 = cvr_list
        assert cvr_1.id == "60001-1-1"
        assert cvr_1.tally_pool == "60001-1"
        assert not cvr_1.pool, f"{cvr_1.pool=}"
        assert list(cvr_1.votes.keys()) == ["111"]
        # Reading Dominion CVRs now takes "IsVote" and "Modified" values into account,
        # so {"6": 1, "1": 2} now becomes {"6": 1} (vote for candidate 1 in this
        # testcase is marked "IsVote": false).  For the same reason, the call to
        # get_vote_for("111", "1") is now False.
        assert cvr_1.votes["111"] == {"6": 1}
        assert cvr_1.get_vote_for("111", "6")
        assert not cvr_1.get_vote_for("111", "1")
        assert not cvr_1.get_vote_for("111", "999")
        assert cvr_2.id == "60009-3-21"
        assert cvr_2.tally_pool == "60009-3"
        assert not cvr_2.pool, f"{cvr_2.pool=}"
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
            "tests/core/data/Dominion_CVRs/test_5.10.50.85.Dominion.json",
            use_current=False,
            enforce_rules=False,
        )
        assert len(cvr_list) == 3
        cvr_1, cvr_2, cvr_3 = cvr_list
        assert cvr_1.id == "1-2-13"
        assert cvr_1.tally_pool == "1-2"
        assert list(cvr_1.votes.keys()) == ["1"]
        assert cvr_1.votes["1"] == {"5": 1}
        assert cvr_1.get_vote_for("1", "5")
        assert cvr_1.get_vote_for("1", "999") is False
        assert cvr_2.id == "1-5-119"
        assert cvr_2.tally_pool == "1-5"
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
        )
        assert len(cvr_list) == 3
        cvr_1, cvr_2, cvr_3 = cvr_list
        assert cvr_1.id == "1-2-13"
        assert cvr_1.tally_pool == "1-2"
        assert not cvr_1.pool
        assert list(cvr_1.votes.keys()) == ["1"]
        assert cvr_1.votes["1"] == {"5": 1}
        assert cvr_1.get_vote_for("1", "5")
        assert cvr_1.get_vote_for("1", "999") is False
        assert cvr_2.id == "1-5-119"
        assert cvr_2.tally_pool == "1-5"
        assert not cvr_2.pool
        assert list(cvr_2.votes.keys()) == ["1"]
        assert cvr_2.votes["1"] == {}
        assert cvr_2.get_vote_for("1", "6") is False
        assert cvr_2.get_vote_for("1", "999") is False
        # CVR 3 tests the updated image_mask regex and the fix for multiple "Cards" entries
        assert cvr_3.id == "1-17-123456789"
        assert list(cvr_3.votes.keys()) == ["12", "13", "19", "20"]
        assert cvr_3.votes["12"] == {"17": 1}
        assert cvr_3.get_vote_for("13", "18") == 1
        assert cvr_3.votes["19"] == {"26": 2, "27": 1}
        assert cvr_3.get_vote_for("20", "192") == 1

    def test_read_cvrs_new_format_vbm_only(self):
        """
        Test reading the newer Dominion CVR format, selecting VBM results only
        """
        cvr_list = Dominion.read_cvrs(
            "tests/core/data/Dominion_CVRs/test_5.10.50.85.Dominion.json",
            use_current=False,
            include_groups=[2],
        )
        assert len(cvr_list) == 1
        cvr_2 = cvr_list[0]
        assert cvr_2.id == "1-5-119"
        assert cvr_2.tally_pool == "1-5"
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
        assert cvr_1.id == "1-2-13"
        assert cvr_1.tally_pool == "1-2"
        assert not cvr_1.pool, f"{cvr_1.pool=}"
        assert list(cvr_1.votes.keys()) == ["1"]
        assert cvr_1.votes["1"] == {"5": 1}
        assert cvr_1.get_vote_for("1", "5")
        assert cvr_1.get_vote_for("1", "999") is False
        assert cvr_2.id == "1-5-119"
        assert cvr_2.tally_pool == "1-5"
        assert cvr_2.pool, f"{cvr_2.pool=}"
        assert list(cvr_2.votes.keys()) == ["1"]
        # Reading Dominion CVRs now takes "IsVote" and "Modified" values into account,
        # so {"6": 1} now becomes {}.  For the same reason, get_vote_for("1", "6") is
        # now also false
        assert cvr_2.votes["1"] == {}
        assert cvr_2.get_vote_for("1", "6") is False
        assert cvr_2.get_vote_for("1", "999") is False

    def test_sample_from_manifest(self):
        """
        Test the card lookup function
        """
        sample = [1, 99, 100, 101, 121, 200, 201]
        d = [
            {
                "Tray #": 1,
                "Tabulator Number": 17,
                "Batch Number": 1,
                "Total Ballots": 100,
                "VBMCart.Cart number": 1,
            },
            {
                "Tray #": 2,
                "Tabulator Number": 18,
                "Batch Number": 2,
                "Total Ballots": 100,
                "VBMCart.Cart number": 2,
            },
            {
                "Tray #": 3,
                "Tabulator Number": 19,
                "Batch Number": 3,
                "Total Ballots": 100,
                "VBMCart.Cart number": 3,
            },
        ]
        manifest = pd.DataFrame.from_dict(d)
        manifest["cum_cards"] = manifest["Total Ballots"].cumsum()
        cards, sample_order, mvr_phantoms = Dominion.sample_from_manifest(
            manifest, sample
        )
        # cart, tray, tabulator, batch, card in batch, imprint, absolute index
        print(f"{cards=}")
        assert cards[0] == [1, 1, 17, 1, 1, "17-1-1", 1]
        assert cards[1] == [1, 1, 17, 1, 99, "17-1-99", 99]
        assert cards[2] == [1, 1, 17, 1, 100, "17-1-100", 100]
        assert cards[3] == [2, 2, 18, 2, 1, "18-2-1", 101]
        assert cards[4] == [2, 2, 18, 2, 21, "18-2-21", 121]
        assert cards[5] == [2, 2, 18, 2, 100, "18-2-100", 200]
        assert cards[6] == [3, 3, 19, 3, 1, "19-3-1", 201]
        assert len(mvr_phantoms) == 0

    def test_sample_from_cvrs(self):
        """
        Test sampling from a list of CVRs using a manifest
        """

        # Construct a list of CVRs and a corresponding manifest
        cvr_list = []
        mnf_list = []
        tray = 1
        batch = 100
        cards_per_batch = 20
        for tab in range(10, 13):
            for batch in range(100, 105):
                mnf_list.append(
                    {
                        "Tray #": tray,
                        "Tabulator Number": tab,
                        "Batch Number": batch,
                        "Total Ballots": cards_per_batch,
                        "VBMCart.Cart number": 1,
                    },                    
                )
                for card in range(1, cards_per_batch + 1):
                    id = f"{tab}-{batch}-{card}"
                    cvr_list.append(CVR(id=id, card_in_batch=card))
            tray += 1

        manifest = pd.DataFrame.from_dict(mnf_list)
        manifest["cum_cards"] = manifest["Total Ballots"].cumsum()
        manifest, _, _ = Dominion.prep_manifest(manifest, 300, len(cvr_list))
        
        # Draw the sample
        sample = [2, 22, 25, 78, 151, 191, 196, 203, 233, 254]
        cards, sample_order, cvr_sample, mvr_phantoms = Dominion.sample_from_cvrs(cvr_list, manifest, sample)
        _cards = sorted([card[5] for card in cards])
        _cvrs = sorted([cvr.id for cvr in cvr_sample])
        _order = sorted(sample_order.keys())
        assert len(mvr_phantoms) == 0
        assert sorted(_cards) == sorted(_cvrs)
        assert sorted(_order) == sorted(_cvrs)
        assert sorted(set(_cvrs)) == [
            '10-100-3',
            '10-101-3',
            '10-101-6',
            '10-103-19',
            '11-102-12',
            '11-104-12',
            '11-104-17',
            '12-100-4',
            '12-101-14',
            '12-102-15',
        ]


##########################################################################################
if __name__ == "__main__":
    sys.exit(pytest.main(["-qq"], plugins=None))
