"""
Tools to read and parse Dominion ballot manifests and CVRs
"""

import json
import numpy as np
import csv
import warnings
import re
import glob
from collections.abc import Collection
from collections import defaultdict
from shangrla.core.Audit import Audit, CVR
from shangrla.core.NonnegMean import NonnegMean


class Dominion:

    @classmethod
    def prep_manifest(cls, manifest, max_cards, n_cvrs):
        """
        Prepare a Dominion Excel ballot manifest (read as a pandas dataframe) for sampling.
        The manifest may have cards that do not contain the contest, but every listed CVR
        is assumed to have a corresponding card in the manifest.

        If the number of cards in the manifest is less than the number of cards that might have been cast,
        max_cards, an additional batch of "phantom" cards is appended to the manifest to make up the difference.

        Parameters:
        ----------
        manifest: dataframe
            should contain the columns
               'Tray #', 'Tabulator Number', 'Batch Number', 'Total Ballots', 'VBMCart.Cart number'
        max_cards: int
            upper bound on the number of cards cast
        n_cvrs: int
            number of CVRs

        Returns:
        --------
        manifest: dataframe
            original manifest with additional column for cumulative cards and, if needed, an additional
            batch for any phantom cards
        manifest_cards: int
            the total number of cards in the manifest
        phantoms: int
            the number of phantom cards required
        """
        cols = [
            "Tray #",
            "Tabulator Number",
            "Batch Number",
            "Total Ballots",
            "VBMCart.Cart number",
        ]
        assert set(cols).issubset(manifest.columns), "missing columns"
        manifest_cards = manifest["Total Ballots"].sum()
        assert (
            manifest_cards <= max_cards
        ), f"cards in manifest {manifest_cards} exceeds max possible {max_cards}"
        assert (
            manifest_cards >= n_cvrs
        ), f"number of cvrs {n_cvrs} exceeds number of cards in the manifest {manifest_cards}"
        phantoms = 0
        if manifest_cards < max_cards:
            phantoms = max_cards - manifest_cards
            warnings.warn(
                f"manifest does not account for every card; appending batch of {phantoms} "
                + f"phantom cards to the manifest"
            )
            r = {
                "Tray #": None,
                "Tabulator Number": "phantom",
                "Batch Number": 1,
                "Total Ballots": phantoms,
                "VBMCart.Cart number": None,
            }
            manifest = manifest.append(r, ignore_index=True)
        manifest["cum_cards"] = manifest["Total Ballots"].cumsum()
        for c in ["Tray #", "Tabulator Number", "Batch Number", "VBMCart.Cart number"]:
            manifest[c] = manifest[c].astype(str)
        return manifest, manifest_cards, phantoms

    @classmethod
    def read_cvrs(
        cls,
        cvr_file: str,
        use_current: bool = True,
        enforce_rules: bool = True,
        include_groups: Collection = [],
        pool_groups: Collection = [],
    ):
        """
        Read CVRs in Dominion format.
        Dominion uses:
           "Id" as the card ID
           "Marks" as the container for votes
           "Rank" as the rank

        Parameters:
        -----------
        cvr_file: string
            filename for cvrs
        use_current: bool [optional], default True
            if set, ignores votes unless `IsCurrent == True`
        enforce_rules: bool [optional], default True
            if set, ignores votes unless `IsVote == True`
        include_groups: enumerable
            if nonempty, use to select only CVRs with specified "CountingGroupId", e.g. (2,) for VBM
        pool_groups: enumerable
            if nonempty, CVRs with `CountingGroupId` in any of the groups is labeled as pooled (for ONEAudit)
            for subsequent construction of ONEAudit CVRs based on aggregating within each `tally_pool` with
            that `CountingGroupId`.

        Returns:
        --------
        cvr_list: list of CVR objects, with additional fields, viz,
                  `id=str(c["TabulatorId"]) + "_" + str(c["BatchId"])  + "_"  + str(record_id)`
                  `tally_pool=str(c["TabulatorId"]) + "_" + str(c["BatchId"])`
                  `pool=(c["CountingGroupId"] in pool_groups)`

        """
        # Image mask is used if the RecordId has been obfuscated (see below)
        image_mask_pattern = re.compile(r"[0-9]{5}_[0-9]{5}_[0-9]*")

        with open(cvr_file, "r") as f:
            cvr_json = json.load(f)
        # Dominion export wraps the CVRs under several layers; unwrap
        # Desired output format is
        # {"ID": "A-001-01", "votes": {"mayor": {"Alice": 1, "Bob": 2, "Candy": 3, "Dan": 4}}}
        cvr_list = []
        for c in cvr_json["Sessions"]:
            votes = {}
            # Skip CVRs not in the desired include_group (if set)
            if include_groups and c["CountingGroupId"] not in include_groups:
                continue
            # Use adjudicated/updated CVR data (if present and requested)
            for k in [
                j
                for j in c.keys()
                if j in (["Original", "Modified"] if use_current else ["Original"])
            ]:
                # Dominion somewhere between 5.2.18.2 and 5.10.50.85 added another hierarchical level, "Cards"
                if "Cards" in c[k].keys():
                    # List comprehension to combine a list of lists of contests, which is essentially
                    # the contents of c[k]["Cards"][0:n]["Contests"], where n is the number of "Card" entries
                    _selector = [
                        _con
                        for _eachlist in [_c["Contests"] for _c in c[k]["Cards"]]
                        for _con in _eachlist
                    ]
                else:
                    _selector = c[k]["Contests"]
                for con in _selector:
                    contest_votes = {}
                    for mark in con["Marks"]:
                        contest_votes[str(mark["CandidateId"])] = (
                            mark["Rank"] if (mark["IsVote"] or not enforce_rules) else 0
                        )
                    votes[str(con["Id"])] = contest_votes
            # If RecordId is obfuscated, extract it from the ImageMask
            record_id = c["RecordId"]
            if record_id == "X":
                image_match = image_mask_pattern.search(c["ImageMask"])
                if image_match is not None:
                    record_id = int(image_match.group(0).split("_")[-1])
            cvr_list.append(
                CVR(
                    id=str(c["TabulatorId"])
                    + "_"
                    + str(c["BatchId"])
                    + "_"
                    + str(record_id),
                    tally_pool=str(c["TabulatorId"]) + "_" + str(c["BatchId"]),
                    pool=(c["CountingGroupId"] in pool_groups),
                    votes=votes,
                )
            )
        return cvr_list

    @classmethod
    def read_cvrs_directory(
        cls,
        cvr_directory: str,
        use_current: bool = True,
        enforce_rules: bool = True,
        include_groups: Collection = [],
        pool_groups: Collection = [],
    ):
        """
        Read CVRs in Dominion format from a given directory.

        Parameters:
        -----------
        cvr_directory: string
            directory name in which the cvrs can be found
        use_current: bool [optional], default True
            if set, ignores votes unless `IsCurrent == True`
        enforce_rules: bool [optional], default True
            if set, ignores votes unless `IsVote == True`
        include_groups: tuple of ints [optional], default ()
            if set, use to select only CVRs with specified "CountingGroupId" (see read_cvrs)

        Returns:
        --------
        cvr_list: list of CVR objects

        """
        cvr_list = []
        for file in [f for f in sorted(glob.glob(f"{cvr_directory}/CvrExport_*.json"))]:
            cvr_list.extend(
                Dominion.read_cvrs(
                    file, use_current, enforce_rules, include_groups, pool_groups
                )
            )
        return cvr_list

    @classmethod
    def raire_to_dominion(cls, cvr_list: list = None):
        """
        translate raire-style identifiers to Dominion-style identifiers by substituting "-" for "_"

        Parameters
        ----------
        cvr_list: list of CVR objects
            input list

        Returns
        -------
        cvrs: list of CVR objects with "-" substituted for "_" in the id attribute.
        """
        for c in cvr_list:
            c.id = str(c.id).replace("_", "-")
        return cvr_list

    @classmethod
    def get_contest_data(
        contest_manifest,
        candidate_manifest,
        tallies,
        exclude_groups=("WriteIn",),
    ):
        """Extract Contest and Candidate information from the manifest files and
        build a dict of contests which for each contest contains the contest name,
        number of winners, list of candidates (ids and names), list of winners
        (ids and names), and vote and card tallies, looking something like this:

        {
            "1": {
                  "name": "GOVERNER",
                  "n_winners": 1,
                  "social_choice_fn": "PLURALITY",
                  "candidates": [ {"id": "1", "name": "DAVE DAVESON" },
                                  {"id": "2", "name": "ERIC ERICSON" },
                  ],
                  "winners": [ {"id": "2", "ERIC ERICSON"} ],
                  "votes": ["2": 12345, "1": 3721],
                  "cards": 17112,
            },
            "2": {
                  "name": "TREASURER",
                  "n_winners": 1,
                  "social_choice_fn": "PLURALITY",
                  "candidates": [ {"id": "3", "name": "GARY GARYSON" },
                                  {"id": "4", "name": "MIKE MIKESON" },
                  ],
                  "winners": [ {"id": "3", "name": "GARY GARYSON" } ],
                  "votes": ["3": 6112, "4": 5319],
                  "cards": 11219,
            }
        }
        """

        # Init some counters, dicts, etc.
        candidate_list = []
        candidate_dict = {}
        contest_dict = {}

        # Ingest ContestManifest.json
        with open(contest_manifest) as fp:
            contestdata = json.load(fp)

        # Ingest CandidateManifest.json
        with open(candidate_manifest) as fp:
            candidatedata = json.load(fp)

        # Build a list of candidate dicts containing candidate id, candidate name and contest id,
        # along with a dict mapping candidate id to name, making lookup by id easier
        for tuple in [
            (str(c["Id"]), c["Description"], str(c["ContestId"]))
            for c in candidatedata["List"]
            if c["Type"] not in exclude_groups
        ]:
            candidate_list.append(
                {"id": tuple[0], "name": tuple[1], "contest": tuple[2]}
            )
            candidate_dict[tuple[0]] = tuple[1]

        # Now build a list of contests by contest id, each populating the contest name, number of
        # winners, social choice function, and candidate list to start
        for tuple in [
            (str(c["Id"]), c["Description"], c["VoteFor"], c["NumOfRanks"])
            for c in contestdata["List"]
        ]:
            contest_dict[tuple[0]] = {
                "name": tuple[1],
                "n_winners": tuple[2],
                "social_choice_fn": (
                    Contest.SOCIAL_CHOICE_FUNCTION.PLURALITY
                    if tuple[3] == 0
                    else Contest.SOCIAL_CHOICE_FUNCTION.IRV
                ),
                "candidates": [
                    {"id": cn["id"], "name": cn["name"]}
                    for cn in candidate_list
                    if cn["contest"] == tuple[0]
                ],
            }

        # Now use the provided tallies to determine the winning candidates in each contest,
        # as well as the number of ballot cards containing each contest

        votes = tallies["votes"]
        cards = tallies["cards"]

        for c in contest_dict.keys():
            try:
                v = votes[c].items()
                v = dict(sorted(v, key=lambda item: item[1], reverse=True))
            except KeyError:
                v = {}
            contest_dict[c]["winners"] = [
                {"id": n, "name": candidate_dict[n]}
                for n in list(v.keys())[: contest_dict[c]["n_winners"]]
            ]
            contest_dict[c]["votes"] = v
            try:
                contest_dict[c]["cards"] = cards[c]
            except KeyError:
                contest_dict[c]["cards"] = 0

        return contest_dict

    @classmethod
    def make_contest(
        id,
        data,
        risk_limit: float=0.05,
        assertion_files={},
        audit_type=Audit.AUDIT_TYPE.CARD_COMPARISON,
        test=NonnegMean.alpha_mart,
        estim=NonnegMean.optimal_comparison,
    ):
        """Returns a contest dict suitable for SHANGRLA"""
        return {
            "name": data["name"],
            "risk_limit": risk_limit,
            "cards": data["cards"],
            "choice_function": data["social_choice_fn"],
            "n_winners": data["n_winners"],
            "candidates": [c["id"] for c in data["candidates"]],
            "winner": [c["id"] for c in data["winners"]],
            "assertion_file": (
                assertion_files[id]
                if id in assertion_files
                and data["social_choice_fn"] == Contest.SOCIAL_CHOICE_FUNCTION.IRV
                else None
            ),
            "audit_type": audit_type,
            "test": test,
            "estim": estim,
        }

    @classmethod
    def generate_contest_dict(
        cvr_list,
        contest_manifest,
        candidate_manifest,
        assertion_files,
    ):
        """
        Given a list of CVRs, a contest manifest and a candidate manifest, and optionally a dict
        of assertion files (where the key is the contest ID and the value, a filename, is a string),
        return a dict of contests ready to process with SHANGRLA
        """
    
        # First, get the vote and card tallies from the CVR list.  The returned value is a dict of
        # dicts with two top level keys "votes" and "cards".
        tallies = tally_cvrs(cvr_list)

        # Next, pull the candidate/contest data together and use the tallies to determine each
        # contest winner
        cdict = get_contest_data(
            contest_manifest,
            candidate_manifest,
            tallies,
            exclude_groups=(),
        )

        # Finally, use the consolidated contest/candidate data to generate the contest dicts
        # ready for SHANGRLA
        c = {}
        for k, v in cdict.items():
            c[k] = make_contest(k, v, risk_limit=0.05, assertion_files=assertion_files)

        return c

    @classmethod
    def sample_from_manifest(cls, manifest, sample):
        """
        Sample from the ballot manifest. Assumes manifest has been augmented to include phantoms.
        Create list of sampled cards, with identifiers.
        Create mvrs for sampled phantoms.

        Parameters
        ----------
        manifest: dataframe
            the processed Dominion manifest, including phantom batches if max_cards exceeds the
            number of cards in the original manifest
        sample: list of ints
            the cards to sample

        Returns
        -------
        cards: list
            sorted list of card identifiers corresponding to the sample. Card identifiers are 1-indexed.
            Each sampled card is listed as
                cart number, tray number, tabulator number, batch, card in batch, tabulator+batch+card_in_batch
        sample_order: dict
            keys are card identifiers, values are dicts containing keys for "selection_order" and "serial"
        mvr_phantoms: list
            list of mvrs for sampled phantoms. The id for the mvr is 'phantom-' concatenated with the row.
        """
        cards = []
        sample_order = {}
        mvr_phantoms = []
        lookup = np.array([0] + list(manifest["cum_cards"]))
        for i, s in enumerate(sample):
            batch_num = int(np.searchsorted(lookup, s, side="left"))
            card_in_batch = int(s - lookup[batch_num - 1])
            tab = manifest.iloc[batch_num - 1]["Tabulator Number"]
            batch = manifest.iloc[batch_num - 1]["Batch Number"]
            card_id = f"{tab}-{batch}-{card_in_batch}"
            card = list(
                manifest.iloc[batch_num - 1][["VBMCart.Cart number", "Tray #"]]
            ) + [tab, batch, card_in_batch, card_id, s]
            cards.append(card)
            if tab == "phantom":
                mvr_phantoms.append(CVR(id=card_id, votes={}, phantom=True))
            sample_order[card_id] = {}
            sample_order[card_id]["selection_order"] = i
            sample_order[card_id]["serial"] = s + 1
        cards.sort(key=lambda x: x[-1])
        return cards, sample_order, mvr_phantoms

    @classmethod
    def sample_from_cvrs(cls, cvr_list: list, manifest: list, sample: np.array):
        """
        Sample from a list of CVRs: return info to find the cards, CVRs, & mvrs for sampled phantom cards

        Parameters
        ----------
        cvr_list: list of CVR objects.
            The id for the cvr is assumed to be composed of a scanner number, batch number, and
            ballot number, joined with underscores, Dominion's format

        manifest: pandas dataframe
            a ballot manifest as a pandas dataframe
        sample: numpy array of ints
            the CVRs to sample

        Returns
        -------
        cards: list
            card identifiers corresponding to the sample, sorted by identifier
        sample_order: dict
            keys are card identifiers, values are dicts containing keys for "selection_order" and "serial"
        cvr_sample: list of CVR objects
            the CVRs in the sample
        mvr_phantoms: list of CVR objects
            the mvrs for phantom sheets in the sample
        """
        cards = []
        sample_order = {}
        cvr_sample = []
        mvr_phantoms = []
        for i, s in enumerate(sample):
            cvr_sample.append(cvr_list[s])
            cvr_id = cvr_list[s].id
            tab, batch, card_num = cvr_id.split("-")
            card_id = f"{tab}-{batch}-{card_num}"
            if not cvr_list[s].phantom:
                manifest_row = manifest[
                    (manifest["Tabulator Number"] == str(tab))
                    & (manifest["Batch Number"] == str(batch))
                ].iloc[0]
                card = [manifest_row["VBMCart.Cart number"], manifest_row["Tray #"]] + [
                    tab,
                    batch,
                    card_num,
                    card_id,
                ]
            else:
                card = ["", "", tab, batch, card_num, card_id]
                mvr_phantoms.append(CVR(id=cvr_id, votes={}, phantom=True))
            cards.append(card)
            sample_order[card_id] = {}
            sample_order[card_id]["selection_order"] = i
            sample_order[card_id]["serial"] = s + 1
        # sort by id
        cards.sort(key=lambda x: x[5])
        return cards, sample_order, cvr_sample, mvr_phantoms

    @classmethod
    def write_cards_sampled(
        cls, sample_file: str, cards: list, print_phantoms: bool = True
    ):
        """
        Write the identifiers of the sampled CVRs to a file.

        Parameters
        ----------
        sample_file: string
            filename for output

        cards: list of lists
            'VBMCart.Cart number','Tray #','Tabulator Number','Batch Number', 'ballot_in_batch',
                  'imprint', 'absolute_card_index'

        print_phantoms: Boolean
            if print_phantoms, prints all sampled cards, including "phantom" cards that were not in
            the original manifest.
            if not print_phantoms, suppresses "phantom" cards

        Returns
        -------

        """
        with open(sample_file, "a") as f:
            writer = csv.writer(f)
            writer.writerow(
                [
                    "cart",
                    "tray",
                    "tabulator",
                    "batch",
                    "card in batch",
                    "imprint",
                    "absolute card index",
                ]
            )
            if print_phantoms:
                for row in cards:
                    writer.writerow(row)
            else:
                for row in cards:
                    if row[2] != "phantom":
                        writer.writerow(row)
