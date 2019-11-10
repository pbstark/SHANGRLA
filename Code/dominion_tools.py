"""
Tools to read and parse Dominion ballot manifests and CVRs
"""
import json
import numpy as np
import csv
import warnings
from assertion_audit_utils import CVR


def prep_dominion_manifest(manifest, N_ballots):
    """
    Prepare a Dominion Excel ballot manifest (read as a pandas dataframe) for sampling
    
    Parameters:
    ----------
    manifest : dataframe
        should contain the columns
           'Tray #', 'Tabulator Number', 'Batch Number', 'Total Ballots', 'VBMCart.Cart number'
    """
    cols = ['Tray #', 'Tabulator Number', 'Batch Number', 'Total Ballots', 'VBMCart.Cart number']
    assert set(cols).issubset(manifest.columns), "missing columns"
    tot_ballots = manifest['Total Ballots'].sum()
    assert tot_ballots <= N_ballots, "Manifest has more ballots than were cast"
    if tot_ballots < N_ballots:
        warnings.warn('Manifest does not account for every ballot cast')
    manifest['cum_ballots'] = manifest['Total Ballots'].cumsum()   

def read_dominion_cvrs(cvr_file):
    """
    Read CVRs in Dominion format.
    Dominion uses:
       "Id" as the ballot ID
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
    # Desired format is
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
        the ballots to sample    
        
    Returns:
    -------
    sorted list of ballot identifiers corresponding to the sample. Ballot identifiers are zero-indexed
    """
    sam = np.sort(sample)
    ballots = []
    lookup = np.array([0] + list(manifest['cum_ballots']))
    for s in sam:
        batch_num = np.searchsorted(lookup, s+1, side='left')
        ballot_in_batch = s-lookup[batch_num-1]+1
        tab = manifest.iloc[batch_num-1]['Tabulator Number']
        batch = manifest.iloc[batch_num-1]['Batch Number']
        ballot = list(manifest.iloc[batch_num-1][['VBMCart.Cart number','Tray #']]) \
                + [tab, batch, ballot_in_batch, str(tab)+'-'+str(batch)+'-'+str(ballot_in_batch), s+1]
        ballots.append(ballot)
    return ballots


def write_ballots_sampled(sample_file, ballots):
    """
    Write the identifiers of the sampled ballots to a file.
    
    Parameters:
    ----------
    
    sample_file : string
        filename for output
        
    ballots : list of lists
        'VBMCart.Cart number','Tray #','Tabulator Number','Batch Number', 'ballot_in_batch', 
              'imprint', 'absolute_ballot_index'
    
    Returns:
    --------
    
    """
    
    with open(sample_file, 'a') as f:
        writer = csv.writer(f)
        writer.writerow(["cart", "tray", "tabulator", "batch",\
                         "ballot in batch", "imprint", "absolute ballot index"])
        for row in ballots:
            writer.writerow(row)

