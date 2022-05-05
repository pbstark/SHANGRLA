"""
Tools to read and parse Hart Intercivic CVRs
"""
import os
import re
import numpy as np
import math
import random
import csv
import pandas as pd
import warnings
import copy
import xml.etree.ElementTree as ET
import xml.dom.minidom
from assertion_audit_utils import \
    Assertion, Assorter, CVR, TestNonnegMean, check_audit_parameters, find_margins,\
    find_p_values, find_sample_size, new_sample_size, summarize_status,\
    write_audit_parameters


def prep_manifest(manifest, max_cards, n_cvrs):
    """
    Prepare a HART Excel ballot manifest (read as a pandas dataframe) for sampling.
    The manifest may have cards that do not contain the contest, but every listed CVR
    is assumed to have a corresponding card in the manifest.

    If the number of cards in the manifest is less than the number of cards that might have been cast,
    max_cards, an additional batch of "phantom" cards is appended to the manifest to make up the difference.

    Parameters:
    ----------
    manifest : dataframe
        should contain the columns
           'Container', 'Tabulator', 'Batch Name', 'Number of Ballots'
    max_cards : int
        upper bound on the number of cards cast
    n_cvrs : int
        number of CVRs

    Returns:
    --------
    manifest : dataframe
        original manifest with additional column for cumulative cards and, if needed, an additional batch for any phantom cards
    manifest_cards : int
        the total number of cards in the manifest
    phantoms : int
        the number of phantom cards required
    """
    cols = ['Container', 'Tabulator', 'Batch Name', 'Number of Ballots']
    assert set(cols).issubset(manifest.columns), "missing columns"
    manifest_cards = manifest['Number of Ballots'].sum()
    assert manifest_cards <= max_cards, f"cards in manifest {manifest_cards} exceeds max possible {max_cards}"
    assert manifest_cards >= n_cvrs, f"number of cvrs {n_cvrs} exceeds number of cards in the manifest {manifest_cards}"
    phantoms = 0
    if manifest_cards < max_cards:
        phantoms = max_cards-manifest_cards
        warnings.warn(f'manifest does not account for every card; appending batch of {phantoms} phantom cards to the manifest')
        r = {'Container': None, 'Tabulator': 'phantom', 'Batch Name': 1, \
             'Number of Ballots': phantoms}
        manifest = manifest.append(r, ignore_index = True)
    manifest['cum_cards'] = manifest['Number of Ballots'].cumsum()
    for c in ['Container', 'Tabulator', 'Batch Name', 'Number of Ballots']:
        manifest[c] = manifest[c].astype(str)
    return manifest, manifest_cards, phantoms


def read_hart_cvr(cvr_path):
    """
    read a single Hart CVR from XML into python

    Parameters:
    -----------
    cvr_path : string
        file path of a single CVR XML file

    Returns:
    --------
    list with seven elements:
        votes: a dict with keys as names of contests and values as the votes in that contest
        undervotes: a list of length len(votes), tabulating the number of undervotes in each contest
        batch_sequence: a string with the batch sequence
        sheet_number: a string with the sheet number of the ballot card
        precinct_name: a string with the precinct name
        precinct_ID: a string with the precinct ID
        cvr_guid: a string of CvrGuide
    """
    namespaces = {'xsi': "http://www.w3.org/2001/XMLSchema-instance",
              "xsd": "http://www.w3.org/2001/XMLSchema",
              "xmlns": "http://tempuri.org/CVRDesign.xsd"}
    #pat = re.compile('[^\w\-\<\>\[\]"\\\']+') # match "word" characters and hyphens
    pat = re.compile('\n/ +/')
    with open(cvr_path, 'r', encoding='latin-1') as xml_file:
        raw_string = xml_file.read()
    cleaned_string = re.sub(pat, " ", raw_string)
    cvr_root = ET.fromstring(cleaned_string)
    batch_sequence = cvr_root.findall("xmlns:BatchSequence", namespaces)[0].text
    sheet_number = cvr_root.findall("xmlns:SheetNumber", namespaces)[0].text
    precinct_name = cvr_root.findall("xmlns:PrecinctSplit", namespaces)[0][0].text
    precinct_ID = cvr_root.findall("xmlns:PrecinctSplit", namespaces)[0][1].text
    cvr_guid = cvr_root.findall("xmlns:CvrGuid", namespaces)[0].text
    contests = []
    votes = {}
    undervotes = []
    #contests are contained in "Contests", the first element of cvr_root, loop through each contest
    for contest in cvr_root[0]:
        #record the name of the contest
        contests.append(contest.findall("xmlns:Name", namespaces)[0].text)
        #check if there are any undervotes. If so, record how many. If not, record 0.
        if contest.findall("xmlns:Undervotes", namespaces):
            undervotes.append(contest.findall("xmlns:Undervotes", namespaces)[0].text)
        else:
            undervotes.append('0')
        #check if there are any valid votes in the contest. If so, record them. If not (only undervotes), record NA.
        if contest.findall("xmlns:Options", namespaces)[0]:
            #initialize dict value as a list, then append the options to it
            if contest.findall("xmlns:Name", namespaces)[0].text not in votes:
                votes[contest.findall("xmlns:Name", namespaces)[0].text] = []
            for options in contest.findall("xmlns:Options", namespaces)[0]:
                #this is for catching write ins, which do not have a "Name" node.
                try:
                    votes[contest.findall("xmlns:Name", namespaces)[0].text].append(options.findall("xmlns:Name", namespaces)[0].text)
                except IndexError:
                    votes[contest.findall("xmlns:Name", namespaces)[0].text].append("WriteIn")

        else:
            votes[contest.findall("xmlns:Name", namespaces)[0].text] = ["NA"]
        # reformat votes to be proper CVR format
        vote_dict = {}
        for key in votes.keys():
            vote_dict[key] = {candidate : True for candidate in votes[key]}

    return CVR(id = batch_sequence + "_" + sheet_number, votes = vote_dict)


def read_cvrs(cvr_folder):
    """
    read a batch of Hart CVRs from XML to list

    Parameters:
    -----------
    cvr_folder : string
        name of folder containing CVRs as XML files

    Returns:
    --------
    cvr_list : list of CVRs as returned by read_hart_CVR()
    """
    cvr_files = os.listdir(cvr_folder)
    cvr_list = []
    for file in cvr_files:
        cvr_list.append(read_hart_cvr(cvr_folder + "/" + file))

    return cvr_list


def sample_from_manifest(manifest, sample):
    """
    Sample from the ballot manifest. Assumes manifest has been augmented to include phantoms.
    Create list of sampled cards, with identifiers.
    Create mvrs for sampled phantoms.

    Parameters
    ----------
    manifest : dataframe
        the processed HART manifest, including phantom batches if max_cards exceeds the
        number of cards in the original manifest
    sample : list of ints
        the cards to sample

    Returns
    -------
    cards : list
        sorted list of card identifiers corresponding to the sample. Card identifiers are 1-indexed.
        Each sampled card is listed as
            cart number, tray number, tabulator number, batch, card in batch, tabulator+batch+card_in_batch
    sample_order : dict
        keys are card identifiers, values are dicts containing keys for "selection_order" and "serial_number"
        Example: {'999
    mvr_phantoms : list
        list of mvrs for sampled phantoms. The id for the mvr is 'phantom-' concatenated with the row.
    """
    cards = []
    sample_order = {}
    mvr_phantoms = []
    lookup = np.array([0] + list(manifest['cum_cards']))
    for i,s in enumerate(sample-1):
        batch_num = int(np.searchsorted(lookup, s, side='left'))
        card_in_batch = int(s-lookup[batch_num-1])
        tab = manifest.iloc[batch_num-1]['Tabulator']
        batch = manifest.iloc[batch_num-1]['Batch Name']
        card_id = f'{tab}-{batch}-{card_in_batch}'
        card = list(manifest.iloc[batch_num-1][['Container']]) \
                + [tab, batch, card_in_batch, card_id]
        cards.append(card)
        if tab == 'phantom':
            mvr_phantoms.append(CVR(id=card_id, votes={}, phantom=True))
        sample_order[card_id] = {}
        sample_order[card_id]["selection_order"] = i
        sample_order[card_id]["serial"] = s+1
    cards.sort(key=lambda x: x[-2])
    return cards, sample_order, mvr_phantoms


def check_for_contest(cvr, contest_name):
    """
    check if a single cvr contains a given contest

    Parameters:
    -----------
    cvr : list
        a single CVR
    contest_name: string
        name of contest

    Returns:
    --------
    contest_present : boolean
        whether contest is present in the CVR
    """
    if contest_name in cvr.votes.keys():
        contest_present = True
    else:
        contest_present = False
    return contest_present

def filter_cvr_contest(cvr_list, contest_name):
    """
    check if a single cvr contains a given contest

    Parameters:
    -----------
    cvr_list : list
        a list of CVRs
    contest_name: string
        name of contest to filter by

    Returns:
    --------
    filtered_cvr_list : list
        CVRs containing contest_name
    """
    filtered_list = list(filter(lambda cvr: check_for_contest(cvr, contest_name), cvr_list))
    return filtered_list


def tabulate_styles(cvr_list):
    """
    tabulate unique CVR styles in cvr_list

    Parameters:
    -----------
    cvr_list: a list of CVRs with dict for contests and votes and list for undervotes, as returned by read_hart_CVRs()

    Returns:
    --------
    a list with two elements: a list of dict keys (styles), being the unique CVR styles in cvr_list, and a corresponding list of counts for each unique style (style_counter)
    """
    # iterate through and find all the unique styles
    styles = []
    for cvr in cvr_list:
        style = cvr.votes.keys()
        if style not in styles:
            styles.append(style)
    # then count the number of times each style appears
    style_counter = [0] * len(styles)
    for cvr in cvr_list:
        style = cvr.votes.keys()
        style_counter[styles.index(style)] += 1
    return [styles, style_counter]


def get_votes(cvr_list):
    """
    tabulate total votes for each candidate in each contest in cvr_list

    Parameters:
    -----------
    cvr_list: a list of CVRs with dict for contests and votes and list for undervotes, as returned by read_hart_CVRs()

    Returns:
    --------
    a dataframe with three columns: the contest, the vote in that contest and the number of votes for that vote in that contest
    """
    styles = tabulate_styles(cvr_list)
    all_votes = pd.DataFrame()
    for i in range(len(cvr_list)):
        vote_df = pd.DataFrame()
        #TODO: BE SURE THIS PROPERLY ACCOUNTS FOR CONTESTS THAT ARE VOTE-FOR-MULTIPLE
        vote_df["contest"] = list(cvr_list[i].votes.keys())
        vote_df["vote"] = [list(item.keys())[0] for item in cvr_list[i].votes.values()]
        vote_df["style"] = styles[0].index(cvr_list[i].votes.keys()) + 1
        all_votes = all_votes.append(vote_df,
                                    ignore_index = True)
    vote_count_df = all_votes.groupby(["contest", "vote", "style"]).size().reset_index().rename(columns = {0 : "num_votes"})
    return vote_count_df

