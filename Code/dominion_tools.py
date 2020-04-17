"""
Tools to read and parse Dominion ballot manifests and CVRs
"""
import json
import numpy as np
import csv
import pandas as pd
import warnings
import copy
from assertion_audit_utils import CVR


def prep_dominion_manifest(manifest, N_cards, n_cvrs):
    """
    Prepare a Dominion Excel ballot manifest (read as a pandas dataframe) for sampling.
    The manifest may have cards that do not contain the contest, but every listed CVR
    is assumed to have a corresponding card in the manifest.

    NOTE: This function has been edited to cater for special primaries manifest style.
    
    If the number of CVRs n_cvrs is less than the number of cards that might contain the contest,
    N_cards, an additional "phantom" batch is added to the manifest to make up the difference.
    
    Parameters:
    ----------
    manifest : dataframe
        should contain the columns
           'Tray #', 'Tabulator Number', 'Batch Number', 'Total Ballots', 'VBMCart.Cart number'
    
    Returns:
    --------
    manifest with additional column for cumulative cards and, if needed, an additional 
    batch for any phantom cards
    
    manifest_cards : the total number of cards in the manifest
    phantom_cards : the number of phantom cards required
    """
    cols = ['State', 'Tabulator ID', 'Batch ID', 'Ballot Count', 'Transfer Case #']
    assert set(cols).issubset(manifest.columns), "missing columns"
    manifest_cards = manifest['Ballot Count'].sum()
    if n_cvrs < N_cards:
        warnings.warn('The CVR list does not account for every card cast in the contest; adding a phantom batch to the manifest')
        r = {'State': None, 'Tabulator ID': 'phantom', 'Batch ID': 1, \
             'Ballot Count': N_cards-n_cvrs, 'Transfer Case #': None}
        manifest = manifest.append(r, ignore_index = True)
    manifest['cum_cards'] = manifest['Ballot Count'].cumsum()    
    for c in ['State', 'Tabulator ID', 'Batch ID', 'Transfer Case #']:
        manifest[c] = manifest[c].astype(str)
    return manifest, manifest_cards, N_cards - n_cvrs

def read_dominion_cvrs(cvr_file):
    """
    Read CVRs in Dominion format.
    Dominion uses:
       "Id" as the card ID
       "Marks" as the container for votes
       "Rank" as the rank
   
    We want to keep group 2 only (VBM)
    
    Parameters:
    -----------
    cvr_file : string
        filename for cvrs
        
    Returns:
    --------
    cvr_list : list of CVR objects
       
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
        cvr_list.append(CVR(ID = str(c["TabulatorId"])\
                                 + '_' + str(c["BatchId"]) \
                                 + '_' + str(c["RecordId"]),\
                                 votes = votes))
    return cvr_list

def sample_from_manifest(manifest, sample):
    """
    Sample from the ballot manifest
    
    Parameters:
    -----------
    manifest : dataframe
        the processed Dominion manifest
    sample : list of ints
        the cards to sample    
        
    Returns:
    -------
    sorted list of card identifiers corresponding to the sample. Card identifiers are 1-indexed
    """
    cards = []
    lookup = np.array([0] + list(manifest['cum_cards']))
    for s in sample:
        batch_num = int(np.searchsorted(lookup, s, side='left'))
        card_in_batch = int(s-lookup[batch_num-1])
        tab = manifest.iloc[batch_num-1]['Tabulator ID']
        batch = manifest.iloc[batch_num-1]['Batch ID']
        card = list(manifest.iloc[batch_num-1][['Transfer Case #','State']]) \
                + [tab, batch, card_in_batch, str(tab)+'-'+str(batch)\
                + '-'+str(card_in_batch), s]
        cards.append(card)
    return cards

def sample_from_cvr(cvr_list, manifest, sample):
    """
    Sample from a list of CVRs. 
    Return information needed to find the corresponding cards, the CVRs in the sample, 
    and a list of mvrs for phantom cards in the sample
    
    Parameters:
    -----------
    cvr_list : list of CVR objects. This function assumes that the id for the cvr is composed
        of a scanner number, batch number, and ballot number, joined with underscores
    manifest : a ballot manifest as a pandas dataframe
    sample : list of ints
        the CVRs to sample    
        
    Returns:
    -------
    cards: sorted list of card identifiers corresponding to the sample.
    cvr_sample: the CVRs in the sample
    mvr_phantoms : list of CVR objects, the mvrs for phantom sheets in the sample
    """
    cards = []
    cvr_sample = []
    mvr_phantoms = []
    
    for s in sample-1:
        cvr_sample.append(cvr_list[s])
        cvr_id = cvr_list[s].id
        tab, batch, card_num = cvr_id.split("-")

        if not cvr_list[s].phantom:
            manifest_row = manifest[(manifest['Tabulator ID'] == str(tab)) \
                                    & (manifest['Batch ID'] == str(batch))].iloc[0]
            card = [manifest_row['Transfer Case #'],\
                    manifest_row['State']] \
                    + [tab, batch, card_num, str(tab)+'-'+str(batch)\
                    + '-'+str(card_num), s]
        else:
            card = ["","", tab, batch, card_num, str(tab)+'-'+str(batch) + '-'+str(card_num), s]
            mvr_phantoms.append(CVR(id=cvr_id, votes = {}, phantom=True))
        cards.append(card)
    # sort by id
    cards.sort(key = lambda x: x[5])
    return cards, cvr_sample, mvr_phantoms

def write_cards_sampled(sample_file, cards, print_phantoms=True):
    """
    Write the identifiers of the sampled CVRs to a file.
    
    Parameters:
    ----------
    
    sample_file : string
        filename for output
        
    cards : list of lists
        'Transfer Case #','State','Tabulator ID','Batch ID', 'ballot_in_batch', 
              'imprint', 'absolute_card_index'
    
    print_phantoms : Boolean
        if print_phantoms, prints all sampled cards, including "phantom" cards that were not in
        the original manifest.
        if not print_phantoms, suppresses "phantom" cards
    
    Returns:
    --------
    
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

def test_sample_from_manifest():
    """
    Test the card lookup function
    """
    sample = [1, 99, 100, 101, 121, 200, 201]
    d = [{'State': 'Kansas', 'Tabulator ID': 17, 'Batch ID': 1, 'Ballot Count': 100, 'Transfer Case #': 'TC-001'},\
        {'State': 'Kansas', 'Tabulator ID': 18, 'Batch ID': 2, 'Ballot Count': 100, 'Transfer Case #': 'TC-002'},\
        {'State': 'Kansas', 'Tabulator ID': 19, 'Batch ID': 3, 'Ballot Count': 100, 'Transfer Case #': 'TC-003'}]
    manifest = pd.DataFrame.from_dict(d)
    manifest['cum_cards'] = manifest['Ballot Count'].cumsum()
    cards = sample_from_manifest(manifest, sample)
    # cart, tray, tabulator, batch, card in batch, imprint, absolute index
    assert cards[0] == ['TC-001', 'Kansas', 17, 1, 1, "17-1-1",1]
    assert cards[1] == ['TC-001', 'Kansas', 17, 1, 99, "17-1-99",99]
    assert cards[2] == ['TC-001', 'Kansas', 17, 1, 100, "17-1-100",100]
    assert cards[3] == ['TC-002', 'Kansas', 18, 2, 1, "18-2-1",101]
    assert cards[4] == ['TC-002', 'Kansas', 18, 2, 21, "18-2-21",121]
    assert cards[5] == ['TC-002', 'Kansas', 18, 2, 100, "18-2-100",200]
    assert cards[6] == ['TC-003', 'Kansas', 19, 3, 1, "19-3-1",201]  

if __name__ == "__main__":
    test_sample_from_manifest()
