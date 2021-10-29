"""
Tools to read and parse Hart Intercivic CVRs
"""
import os
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
    find_p_values, find_sample_size, new_sample_size, prep_sample, summarize_status,\
    write_audit_parameters


def prep_manifest(manifest, N_cards, n_cvrs):
    """
    Prepare a hart ballot manifest (read as a pandas dataframe) for sampling.
    The manifest may have cards that do not contain the contest, but every listed CVR
    is assumed to have a corresponding card in the manifest.

    If the number of CVRs n_cvrs is less than the number of cards that might contain the contest,
    N_cards, an additional "phantom" batch is added to the manifest to make up the difference.

    Parameters:
    ----------
    manifest : dataframe
        should contain the columns
           'Container', 'Tabulator', 'Batch Name', 'Number of Ballots'

    Returns:
    --------
    manifest with additional column for cumulative cards and, if needed, an additional
    batch for any phantom cards

    manifest_cards : the total number of cards in the manifest
    phantom_cards : the number of phantom cards required
    """
    cols = ['Container', 'Tabulator', 'Batch Name', 'Number of Ballots']
    assert set(cols).issubset(manifest.columns), "missing columns"
    manifest_cards = manifest['Number of Ballots'].sum()
    if n_cvrs < N_cards:
        warnings.warn('The CVR list does not account for every card cast in the contest; adding a phantom batch to the manifest')
        r = {'Container': None, 'Tabulator': 'phantom', 'Batch Name': 1, \
             'Number of Ballots': N_cards-n_cvrs}
        manifest = manifest.append(r, ignore_index = True)
    manifest['cum_cards'] = manifest['Number of Ballots'].cumsum()
    for c in ['Container', 'Tabulator', 'Batch Name']:
        manifest[c] = manifest[c].astype(str)
    return manifest, manifest_cards, N_cards - n_cvrs

def read_cvr(cvr_path):
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

    cvr_root = ET.parse(cvr_path).getroot()
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
    #iterate through and find all the unique styles
    styles = []
    for cvr in cvr_list:
        style = cvr.votes.keys()
        if style not in styles:
            styles.append(style)
    #then count the number of times each style appears
    style_counter = [0] * len(styles)
    for cvr in cvr_list:
        style = cvr.votes.keys()
        style_counter[styles.index(style)] += 1
    return [styles, style_counter]

