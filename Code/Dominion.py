"""
Tools to read and parse Dominion ballot manifests and CVRs
"""
import json
import numpy as np
import csv
import pandas as pd
import warnings
import copy
from CVR import CVR

class Dominion:
    
    @classmethod
    def prep_manifest(cls,manifest, max_cards, n_cvrs):
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
            original manifest with additional column for cumulative cards and, if needed, an additional batch for any phantom cards   
        manifest_cards: int
            the total number of cards in the manifest
        phantoms: int
            the number of phantom cards required
        """
        cols = ['Tray #', 'Tabulator Number', 'Batch Number', 'Total Ballots', 'VBMCart.Cart number']
        assert set(cols).issubset(manifest.columns), "missing columns"
        manifest_cards = manifest['Total Ballots'].sum()
        assert manifest_cards <= max_cards, f"cards in manifest {manifest_cards} exceeds max possible {max_cards}"
        assert manifest_cards >= n_cvrs, f"number of cvrs {n_cvrs} exceeds number of cards in the manifest {manifest_cards}"
        phantoms = 0
        if manifest_cards < max_cards:
            phantoms = max_cards-manifest_cards
            warnings.warn(f'manifest does not account for every card; appending batch of {phantoms} phantom cards to the manifest')
            r = {'Tray #': None, 'Tabulator Number': 'phantom', 'Batch Number': 1, \
                 'Total Ballots': phantoms, 'VBMCart.Cart number': None}
            manifest = manifest.append(r, ignore_index = True)
        manifest['cum_cards'] = manifest['Total Ballots'].cumsum()    
        for c in ['Tray #', 'Tabulator Number', 'Batch Number', 'VBMCart.Cart number']:
            manifest[c] = manifest[c].astype(str)
        return manifest, manifest_cards, phantoms

    @classmethod
    def read_cvrs(cls, cvr_file):
        """
        Read CVRs in Dominion format.
        Dominion uses:
           "Id" as the card ID
           "Marks" as the container for votes
           "Rank" as the rank

        We want to keep group 2 only (VBM)

        Parameters:
        -----------
        cvr_file: string
            filename for cvrs

        Returns:
        --------
        cvr_list: list of CVR objects

        """
        with open(cvr_file, 'r') as f:
            cvr_json = json.load(f)
        # Dominion export wraps the CVRs under several layers; unwrap
        # Desired output format is
        # {"ID": "A-001-01", "votes": {"mayor": {"Alice": 1, "Bob": 2, "Candy": 3, "Dan": 4}}}
        cvr_list = []
        for c in cvr_json['Sessions']:
            votes = {}
            for con in c["Original"]["Contests"]:
                contest_votes = {}
                for mark in con["Marks"]:
                    contest_votes[str(mark["CandidateId"])] = mark["Rank"]
                votes[str(con["Id"])] = contest_votes
            cvr_list.append(CVR(id = str(c["TabulatorId"])\
                                     + '_' + str(c["BatchId"]) \
                                     + '_' + str(c["RecordId"]),\
                                     votes = votes))
        return cvr_list

    @classmethod
    def raire_to_dominion(cls, cvr_list: list=None):
        '''
        translate raire-style identifiers to Dominion-style identifiers by substituting "-" for "_"

        Parameters
        ----------
        cvr_list: list of CVR objects
            input list

        Returns
        -------
        cvrs: list of CVR objects with "-" substituted for "_" in the id attribute.
        '''
        for c in cvr_list:
            c.id = str(c.id).replace("_","-")
        return cvr_list

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
        lookup = np.array([0] + list(manifest['cum_cards']))
        for i, s in enumerate(sample):
            batch_num = int(np.searchsorted(lookup, s, side='left'))
            card_in_batch = int(s-lookup[batch_num-1])
            tab = manifest.iloc[batch_num-1]['Tabulator Number']
            batch = manifest.iloc[batch_num-1]['Batch Number']
            card_id = f'{tab}-{batch}-{card_in_batch}'
            card = list(manifest.iloc[batch_num-1][['VBMCart.Cart number','Tray #']]) \
                    + [tab, batch, card_in_batch, card_id, s]
            cards.append(card)
            if tab == 'phantom':
                mvr_phantoms.append(CVR(id=card_id, votes={}, phantom=True))
            sample_order[card_id] = {}
            sample_order[card_id]["selection_order"] = i
            sample_order[card_id]["serial"] = s+1
        cards.sort(key=lambda x: x[-1])
        return cards, sample_order, mvr_phantoms

    @classmethod
    def sample_from_cvrs(cls,cvr_list: list, manifest: list, sample: np.array):
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
        for i,s in enumerate(sample-1):
            cvr_sample.append(cvr_list[s])
            cvr_id = cvr_list[s].id
            tab, batch, card_num = cvr_id.split("-")
            card_id = f'{tab}-{batch}-{card_num}'
            if not cvr_list[s].phantom:
                manifest_row = manifest[(manifest['Tabulator Number'] == str(tab)) \
                                        & (manifest['Batch Number'] == str(batch))].iloc[0]
                card = [manifest_row['VBMCart.Cart number'],\
                        manifest_row['Tray #']] \
                        + [tab, batch, card_num, card_id ]
            else:
                card = ["","", tab, batch, card_num, card_id]
                mvr_phantoms.append(CVR(id=cvr_id, votes = {}, phantom=True))
            cards.append(card)
            sample_order[card_id] = {}
            sample_order[card_id]["selection_order"] = i
            sample_order[card_id]["serial"] = s+1
        # sort by id
        cards.sort(key = lambda x: x[5])
        return cards, sample_order, cvr_sample, mvr_phantoms


    @classmethod
    def write_cards_sampled(cls, sample_file: str, cards: list, print_phantoms: bool=True):
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
        with open(sample_file, 'a') as f:
            writer = csv.writer(f)
            writer.writerow(["cart", "tray", "tabulator", "batch",\
                             "card in batch", "imprint", "absolute card index"])
            if print_phantoms:
                for row in cards:
                    writer.writerow(row) 
            else:
                for row in cards:
                    if row[2] != "phantom":
                        writer.writerow(row) 
