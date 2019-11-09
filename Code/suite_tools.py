from __future__ import print_function, division

from collections import OrderedDict
from itertools import product
import math
import numpy as np
import json
import csv
import matplotlib.pyplot as plt

from ballot_comparison import ballot_comparison_pvalue
from fishers_combination import  maximize_fisher_combined_pvalue, create_modulus
from sprt import ballot_polling_sprt





def write_audit_results(filename, \
                        n1, n2, cvr_sample, nocvr_sample, \
                        o1, o2, u1, u2, observed_poll, \
                        audit_pvalues, prng_state):
    samples = {"cvr_sample" : cvr_sample.tolist(),
               "nocvr_sample" : nocvr_sample.tolist()}
    audit_pvalues_str = {}
    for key, value in audit_pvalues.items():
        audit_pvalues_str[str(key)] = value
    results = {"n1" : int(n1),
               "n2" : int(n2),
               "samples" : samples,
               "o1" : o1,
               "o2" : o2,
               "u1" : u1,
               "u2" : u2,
               "observed_poll" : observed_poll,
               "audit_pvalues" : audit_pvalues_str,
               "prng_state" : prng_state
             }
    with open(filename, 'w') as f:
        json.dump(results, f)






def check_valid_vote_counts(candidates, num_winners, stratum_sizes):
    """
    Check that the candidates dict containing vote totals
    makes sense
    """

    cvr_votes = poll_votes = 0
    for votes in candidates.values():
        cvr_votes += votes[0]
        poll_votes += votes[1]
    assert cvr_votes <= stratum_sizes[0], "More votes entered than total for the stratum"
    assert poll_votes <= stratum_sizes[1], "More votes entered than total for the stratum"
    assert len(candidates) >= num_winners, "Fewer candidates than number of winners"


def check_overvote_rates(margins, total_votes, o1_rate, o2_rate):
    """
    Print a warning if the rates of overvotes that the user supplied
    are large enough to change the outcome.
    """
    o1_num = o1_rate*total_votes
    o2_num = o2_rate*total_votes
    if o1_num > 0.5*max(margins.values()):
        print("Warning: the o1_rate supplied implies that the reported outcome is wrong.")
    if o2_num > 0.25*max(margins.values()):
        print("Warning: the o2_rate supplied implies that the reported outcome is wrong.")

################################################################################
##################### Set up candidate data structures #########################
################################################################################




def print_reported_votes(candidates, winners, losers, margins, stratum_sizes,
                         print_alphabetical=False):
    """
    Utility function to print the contest information

    Parameters
    ----------
    candidates : OrderedDict
        keys are the candidate name, values are a list with
        [reported votes in CVR stratum, reported votes in no-CVR stratum,
        total reported votes]. Sorted in descending order of total votes.
    winners : list
        names of reported winners
    losers : list
        names of reported losers
    margins : OrderedDict
        keys are (reported winner, reported loser), values are the margin
        in votes between the pair. Sorted in descending order of margins.
    stratum_sizes : list
        2 elements, [total votes in CVR stratum, total votes in no-CVR stratum]
    print_alphabetical : bool
        Print candidates in alphabetical order? Default False prints in order
        of reported votes.
    """
    # Sum the votes in each stratum
    cvr_votes = poll_votes = 0
    for votes in candidates.values():
        cvr_votes += votes[0]
        poll_votes += votes[1]

    # Find the smallest margin between winners and losers
    min_margin = np.amin(list(margins.values()))

    # Reorder candidates if print_alphabetical is True
    if print_alphabetical:
        candidates = OrderedDict(sorted(candidates.items()))
    
    tot_valid = cvr_votes + poll_votes
    tot_votes = stratum_sizes[0] + stratum_sizes[1]
    print('\nTotal reported votes:\n\t\t\tCVR\tno-CVR\ttotal\t% of all votes\t% of valid votes')
    for k, v in candidates.items():
        print('\t', k, ':', v[0], '\t', v[1], '\t', v[2], '\t', \
              "{:>2.2%}".format(v[2]/tot_votes), '\t', \
              "{:>2.2%}".format(v[2]/tot_valid))
    print('\n\t valid votes:\t', cvr_votes, \
          '\t', poll_votes, '\t', tot_valid,
          '\t', "{:>2.2%}".format(tot_valid/tot_votes))
    print('\n\t non-votes:\t',\
          stratum_sizes[0] - cvr_votes, '\t',\
          stratum_sizes[1] - poll_votes, '\t',\
          tot_votes - tot_valid,\
          '\t', "{:>2.2%}".format((tot_votes-tot_valid)/tot_votes))

    print('\nReported winners:')
    for w in winners:
        print('\t', w)

    print('\nReported losers:')
    for ell in losers:
        print('\t', ell)

    print('\n\nReported margins:')
    for k, v in margins.items():
        dum = k[0] + ' beat ' + k[1] + ' by'
        print('\t', dum, "{:,}".format(v), 'votes')

    print('\nSmallest reported margin:', "{:,}".format(min_margin), \
          '\nCorresponding reported diluted margin:', "{:.2%}".format(min_margin/np.sum(stratum_sizes)))


################################################################################
########################## Sample size estimation ##############################
################################################################################


def estimate_n(N_w1, N_w2, N_l1, N_l2, N1, N2,\
               o1_rate=0, o2_rate=0, u1_rate=0, u2_rate=0,\
               n_ratio=None,
               risk_limit=0.05,\
               gamma=1.03905,\
               stepsize=0.05,\
               min_n=5,\
               risk_limit_tol=0.8,
               verbose=False):
    """
    Estimate the initial sample sizes for the audit.

    Parameters
    ----------
    N_w1 : int
        votes for the reported winner in the ballot comparison stratum
    N_w2 : int
        votes for the reported winner in the ballot polling stratum
    N_l1 : int
        votes for the reported loser in the ballot comparison stratum
    N_l2 : int
        votes for the reported loser in the ballot polling stratum
    N1 : int
        total number of votes in the ballot comparison stratum
    N2 : int
        total number of votes in the ballot polling stratum
    o1_rate : float
        expected percent of ballots with 1-vote overstatements in
        the CVR stratum
    o2_rate : float
        expected percent of ballots with 2-vote overstatements in
        the CVR stratum
    u1_rate : float
        expected percent of ballots with 1-vote understatements in
        the CVR stratum
    u2_rate : float
        expected percent of ballots with 2-vote understatements in
        the CVR stratum
    n_ratio : float
        ratio of sample allocated to each stratum.
        If None, allocate sample in proportion to ballots cast in each stratum
    risk_limit : float
        risk limit
    gamma : float
        gamma from Lindeman and Stark (2012)
    stepsize : float
        stepsize for the discrete bounds on Fisher's combining function
    min_n : int
        smallest acceptable initial sample size. Default 5
    risk_limit_tol : float
        acceptable percentage below the risk limit, between 0 and 1.
        Default is 0.8, meaning the estimated sample size might have
        an expected risk that is 80% of the desired risk limit
    verbose : bool
        If True, print the sample size and expected p-value at each search step.
        Defaults to False.
    Returns
    -------
    tuple : estimated initial sample sizes in the CVR stratum and no-CVR stratum
    """
    n_ratio = n_ratio if n_ratio else N1/(N1+N2)
    n = min_n
    assert n > 0, "minimum sample size must be positive"
    assert risk_limit_tol < 1 and risk_limit_tol > 0, "bad risk limit tolerance"

    reported_margin = (N_w1+N_w2)-(N_l1+N_l2)
    expected_pvalue = 1

    if N1 == 0:
        def try_n(n):
            n = int(n)
            sample = [0]*math.ceil(n*N_l2/N2)+[1]*int(n*N_w2/N2)
            if len(sample) < n:
                sample += [np.nan]*(n - len(sample))
            expected_pvalue = ballot_polling_sprt(sample=np.array(sample), \
                            popsize=N2, \
                            alpha=risk_limit,\
                            Vw=N_w2, Vl=N_l2, \
                            null_margin=0)['pvalue']
            if verbose:
                print('...trying...', n, expected_pvalue)
            return expected_pvalue

    elif N2 == 0:
        def try_n(n):
            o1 = math.ceil(o1_rate*n)
            o2 = math.ceil(o2_rate*n)
            u1 = math.floor(u1_rate*n)
            u2 = math.floor(u2_rate*n)
            expected_pvalue = ballot_comparison_pvalue(n=n, \
                            gamma=gamma, o1=o1, u1=u1, o2=o2, u2=u2, \
                            reported_margin=reported_margin, N=N1, \
                            null_lambda=1)
            if verbose:
                print('...trying...', n, expected_pvalue)
            return expected_pvalue

    else:
        def try_n(n):
            """
            Find expected combined P-value for a total sample size n.
            """
            n1 = math.ceil(n_ratio * n)
            n2 = int(n - n1)

            # Set up the p-value function for the CVR stratum
            if n1 == 0:
                cvr_pvalue = lambda alloc: 1
            else:
                o1 = math.ceil(o1_rate*n1)
                o2 = math.ceil(o2_rate*n1)
                u1 = math.floor(u1_rate*n1)
                u2 = math.floor(u2_rate*n1)
                cvr_pvalue = lambda alloc: ballot_comparison_pvalue(n=n1, \
                                gamma=gamma, o1=o1, u1=u1, o2=o2, u2=u2, \
                                reported_margin=reported_margin, N=N1, \
                                null_lambda=alloc)

            # Set up the p-value function for the no-CVR stratum
            if n2 == 0:
                nocvr_pvalue = lambda alloc: 1
            else:
                sample = [0]*math.ceil(n2*N_l2/N2)+[1]*int(n2*N_w2/N2)
                if len(sample) < n2:
                    sample += [np.nan]*(n2 - len(sample))
                nocvr_pvalue = lambda alloc: ballot_polling_sprt(sample=np.array(sample), \
                                popsize=N2, \
                                alpha=risk_limit,\
                                Vw=N_w2, Vl=N_l2, \
                                null_margin=(N_w2-N_l2) - \
                                 alloc*reported_margin)['pvalue']

            if N2 == 0:
                n_w2 = 0
                n_l2 = 0
            else:
                n_w2 = int(n2*N_w2/N2)
                n_l2 = int(n2*N_l2/N2)
            bounding_fun = create_modulus(n1=n1, n2=n2,
                                          n_w2=n_w2, \
                                          n_l2=n_l2, \
                                          N1=N1, V_wl=reported_margin, gamma=gamma)
            res = maximize_fisher_combined_pvalue(N_w1=N_w1, N_l1=N_l1, N1=N1, \
                                                  N_w2=N_w2, N_l2=N_l2, N2=N2, \
                                                  pvalue_funs=(cvr_pvalue, \
                                                   nocvr_pvalue), \
                                                  stepsize=stepsize, \
                                                  modulus=bounding_fun, \
                                                  alpha=risk_limit)
            expected_pvalue = res['max_pvalue']
            if verbose:
                print('...trying...', n, expected_pvalue)
            return expected_pvalue

    # step 1: linear search, doubling n each time
    while (expected_pvalue > risk_limit) or (expected_pvalue is np.nan):
        n = 2*n
        if n > N1+N2:
            n1 = math.ceil(n_ratio * (N1+N2))
            n2 = int(N1 + N2 - n1)
            return (n1, n2)
        if N2 > 0:
            n1 = math.ceil(n_ratio * n)
            n2 = int(n - n1)
            if (N_w2 < int(n2*N_w2/N2) or N_l2 < int(n2*N_l2/N2)):
                return(N1, N2)
        expected_pvalue = try_n(n)

    # step 2: bisection between n/2 and n
    low_n = n/2
    high_n = n
    mid_pvalue = 1
    while  (mid_pvalue > risk_limit) or (mid_pvalue < risk_limit_tol*risk_limit) or \
        (expected_pvalue is np.nan):
        mid_n = np.floor((low_n+high_n)/2)
        if (low_n == mid_n) or (high_n == mid_n):
            break
        mid_pvalue = try_n(mid_n)
        if mid_pvalue <= risk_limit:
            high_n = mid_n
        else:
            low_n = mid_n
    
    n1 = math.ceil(n_ratio * high_n)
    n2 = math.ceil(high_n - n1)
    return (n1, n2)


def check_polling_sample_size(candidates, winners, losers, \
                              stratum_sizes, risk_limit):
    """
    Print what the sample size would need to be, assuming everything
    were done using ballot polling
    """
    sample_sizes = {}
    for k in product(winners, losers):
        sample_sizes[k] = estimate_n(N_w1 = 0,\
                                     N_w2 = candidates[k[0]][0]+candidates[k[0]][1],\
                                     N_l1 = 0,\
                                     N_l2 = candidates[k[1]][0]+candidates[k[1]][1],\
                                     N1 = 0,\
                                     N2 = np.sum(stratum_sizes),\
                                     n_ratio = 0,\
                                     risk_limit = risk_limit,\
                                     min_n = 5,\
                                     risk_limit_tol = 0.95)
    sample_size = np.amax([v[0]+v[1] for v in sample_sizes.values()])
    print('\n\nexpected minimum sample size that would be needed using ballot polling ONLY, for all ballots:', sample_size)


def plot_nratio_sample_sizes(candidates, winners, losers, stratum_sizes, \
                             n_ratio_step=0.1, \
                             o1_rate=0, o2_rate=0, u1_rate=0, u2_rate=0, \
                             risk_limit=0.05, gamma=1.03905, stepsize=0.05):
    
    sample_size_estimate = []
    for nratio_val in np.arange(0, 1+n_ratio_step, n_ratio_step):
        print("trying", nratio_val)
        sample_sizes = {}
        for k in product(winners, losers):
            sample_sizes[k] = estimate_n(N_w1 = candidates[k[0]][0],\
                                         N_w2 = candidates[k[0]][1],\
                                         N_l1 = candidates[k[1]][0],\
                                         N_l2 = candidates[k[1]][1],\
                                         N1 = stratum_sizes[0],\
                                         N2 = stratum_sizes[1],\
                                         o1_rate = o1_rate,\
                                         o2_rate = o2_rate,\
                                         u1_rate = u1_rate,\
                                         u2_rate = u2_rate,\
                                         n_ratio = nratio_val,\
                                         risk_limit = risk_limit,\
                                         gamma = gamma,\
                                         stepsize = stepsize,\
                                         min_n = 5,\
                                         risk_limit_tol = 0.95)
        sample_size_estimate.append(np.amax([v[0]+v[1] for v in sample_sizes.values()]))
    plt.plot(np.arange(0, 1+n_ratio_step, n_ratio_step), sample_size_estimate)
    plt.xlabel("fraction of sample drawn from the CVR stratum (n_ratio)")
    plt.ylabel("total sample size")
    plt.show()


def estimate_escalation_n(N_w1, N_w2, N_l1, N_l2, N1, N2, n1, n2, \
                          o1_obs, o2_obs, u1_obs, u2_obs, \
                          n2l_obs, n2w_obs, \
                          o1_rate=0, o2_rate=0, u1_rate=0, u2_rate=0, \
                          n_ratio=None, \
                          risk_limit=0.05,\
                          gamma=1.03905,\
                          stepsize=0.05,\
                          risk_limit_tol=0.8,
                          verbose=False):
    """
    Estimate the initial sample sizes for the audit.

    Parameters
    ----------
    N_w1 : int
        votes for the reported winner in the ballot comparison stratum
    N_w2 : int
        votes for the reported winner in the ballot polling stratum
    N_l1 : int
        votes for the reported loser in the ballot comparison stratum
    N_l2 : int
        votes for the reported loser in the ballot polling stratum
    N1 : int
        total number of votes in the ballot comparison stratum
    N2 : int
        total number of votes in the ballot polling stratum
    n1 : int
        size of sample already drawn in the ballot comparison stratum
    n2 : int
        size of sample already drawn in the ballot polling stratum
    o1_obs : int
        observed number of ballots with 1-vote overstatements in the CVR stratum
    o2_obs : int
        observed number of ballots with 2-vote overstatements in the CVR stratum
    u1_obs : int
        observed number of ballots with 1-vote understatements in the CVR
        stratum
    u2_obs : int
        observed number of ballots with 2-vote understatements in the CVR
        stratum
    n2l_obs : int
        observed number of votes for the reported loser in the no-CVR stratum
    n2w_obs : int
        observed number of votes for the reported winner in the no-CVR stratum
    n_ratio : float
        ratio of sample allocated to each stratum.
        If None, allocate sample in proportion to ballots cast in each stratum
    risk_limit : float
        risk limit
    gamma : float
        gamma from Lindeman and Stark (2012)
    stepsize : float
        stepsize for the discrete bounds on Fisher's combining function
    risk_limit_tol : float
        acceptable percentage below the risk limit, between 0 and 1.
        Default is 0.8, meaning the estimated sample size might have
        an expected risk that is 80% of the desired risk limit
    verbose : bool
        If True, print the sample size and expected p-value at each search step.
        Defaults to False.
    Returns
    -------
    tuple : estimated initial sample sizes in the CVR stratum and no-CVR stratum
    """
    n_ratio = n_ratio if n_ratio else N1/(N1+N2)
    n = n1+n2
    reported_margin = (N_w1+N_w2)-(N_l1+N_l2)
    expected_pvalue = 1

    n1_original = n1
    n2_original = n2
    observed_nocvr_sample = [0]*n2l_obs + [1]*n2w_obs + \
                            [np.nan]*(n2_original-n2l_obs-n2w_obs)

    # Assume o1, o2, u1, u2 rates will be the same as what we observed in sample
    if n1_original != 0:
        o1_rate = o1_obs/n1_original
        o2_rate = o2_obs/n1_original
        u1_rate = u1_obs/n1_original
        u2_rate = u2_obs/n1_original

    if N1 == 0:
        def try_n(n):
            n = int(n)
            expected_new_sample = [0]*math.ceil((n-n2_original)*(n2l_obs/n2_original))+ \
                                  [1]*int((n-n2_original)*(n2w_obs/n2_original))
            totsample = observed_nocvr_sample+expected_new_sample
            if len(totsample) < n:
                totsample += [np.nan]*(n - len(totsample))
            totsample = np.array(totsample)
            n_w2 = np.sum(totsample == 1)
            n_l2 = np.sum(totsample == 0)

            expected_pvalue = ballot_polling_sprt( \
                            sample=totsample,\
                            popsize=N2, \
                            alpha=risk_limit,\
                            Vw=N_w2, Vl=N_l2, \
                            null_margin=0)['pvalue']
            if verbose:
                print('...trying...', n, expected_pvalue)
            return expected_pvalue

    elif N2 == 0:
        def try_n(n):
            o1 = math.ceil(o1_rate*(n-n1_original)) + o1_obs
            o2 = math.ceil(o2_rate*(n-n1_original)) + o2_obs
            u1 = math.floor(u1_rate*(n-n1_original)) + u1_obs
            u2 = math.floor(u2_rate*(n-n1_original)) + u2_obs
            expected_pvalue = ballot_comparison_pvalue(n=n,\
                                    gamma=1.03905, o1=o1, \
                                    u1=u1, o2=o2, u2=u2, \
                                    reported_margin=reported_margin, N=N1, \
                                    null_lambda=1)
            if verbose:
                print('...trying...', n, expected_pvalue)
            return expected_pvalue
    
    else:
        def try_n(n):
            n1 = math.ceil(n_ratio * n)
            n2 = int(n - n1)
        
            if (n1 < n1_original) or (n2 < n2_original):
                return 1

            # Set up the p-value function for the CVR stratum
            if n1 == 0:
                cvr_pvalue = lambda alloc: 1
            else:
                o1 = math.ceil(o1_rate*(n1-n1_original)) + o1_obs
                o2 = math.ceil(o2_rate*(n1-n1_original)) + o2_obs
                u1 = math.floor(u1_rate*(n1-n1_original)) + u1_obs
                u2 = math.floor(u2_rate*(n1-n1_original)) + u2_obs
                cvr_pvalue = lambda alloc: ballot_comparison_pvalue(n=n1,\
                                    gamma=1.03905, o1=o1, \
                                    u1=u1, o2=o2, u2=u2, \
                                    reported_margin=reported_margin, N=N1, \
                                    null_lambda=alloc)

            # Set up the p-value function for the no-CVR stratum
            if n2 == 0:
                nocvr_pvalue = lambda alloc: 1
                n_w2 = 0
                n_l2 = 0
            else:
                expected_new_sample = [0]*math.ceil((n2-n2_original)*(n2l_obs/n2_original))+ \
                                      [1]*int((n2-n2_original)*(n2w_obs/n2_original))
                totsample = observed_nocvr_sample+expected_new_sample
                if len(totsample) < n2:
                    totsample += [np.nan]*(n2 - len(totsample))
                totsample = np.array(totsample)
                n_w2 = np.sum(totsample == 1)
                n_l2 = np.sum(totsample == 0)

                nocvr_pvalue = lambda alloc: ballot_polling_sprt( \
                                sample=totsample,\
                                popsize=N2, \
                                alpha=risk_limit,\
                                Vw=N_w2, Vl=N_l2, \
                                null_margin=(N_w2-N_l2) - \
                                 alloc*reported_margin)['pvalue']

            # Compute combined p-value
            bounding_fun = create_modulus(n1=n1, n2=n2,
                                          n_w2=n_w2, \
                                          n_l2=n_l2, \
                                          N1=N1, V_wl=reported_margin, gamma=gamma)
            res = maximize_fisher_combined_pvalue(N_w1=N_w1, N_l1=N_l1, N1=N1, \
                                                  N_w2=N_w2, N_l2=N_l2, N2=N2, \
                                                  pvalue_funs=(cvr_pvalue,\
                                                    nocvr_pvalue), \
                                                  stepsize=stepsize, \
                                                  modulus=bounding_fun, \
                                                  alpha=risk_limit)
            expected_pvalue = res['max_pvalue']
            if verbose:
                print('...trying...', n, expected_pvalue)
            return expected_pvalue

    # step 1: linear search, increasing n by a factor of 1.1 each time
    while (expected_pvalue > risk_limit) or (expected_pvalue is np.nan):
        n = np.ceil(1.1*n)
        if n > N1+N2:
            n1 = math.ceil(n_ratio * (N1+N2))
            n2 = int(N1 + N2 - n1)
            return (n1, n2)
        if N2 > 0:
            n1 = math.ceil(n_ratio * n)
            n2 = int(n - n1)
            if (N_w2 < int(n2*N_w2/N2) or N_l2 < int(n2*N_l2/N2)):
                return(N1, N2)
        expected_pvalue = try_n(n)

    # step 2: bisection between n/1.1 and n
    low_n = n/1.1
    high_n = n
    mid_pvalue = 1
    while  (mid_pvalue > risk_limit) or (mid_pvalue < risk_limit_tol*risk_limit) or \
        (expected_pvalue is np.nan):
        mid_n = np.floor((low_n+high_n)/2)
        if (low_n == mid_n) or (high_n == mid_n):
            break
        mid_pvalue = try_n(mid_n)
        if mid_pvalue <= risk_limit:
            high_n = mid_n
        else:
            low_n = mid_n

    n1 = math.ceil(n_ratio * high_n)
    n2 = math.ceil(high_n - n1)
    return (n1, n2)


################################################################################
########################## Ballot manifest tools ###############################
################################################################################

def read_manifest_from_csv(filename):
    """
    Read the ballot manifest into a list in the format ['batch id : number of ballots']
    from CSV file named filename
    """
    manifest = []
    with open(filename, newline='') as csvfile:
        reader = csv.reader(csvfile, delimiter = ",")
        for row in reader:
#            row.remove(row[1])
            batch = " , ".join(row)
            manifest.append(batch)
    return manifest[1:]


def parse_manifest(manifest):
    """
    Parses a ballot manifest.
    Identifiers are not necessarily unique *across* batches.

    Input
    -----
    a ballot manifest in the syntax described above

    Returns
    -------
    an ordered dict containing batch ID (key) and ballot identifiers within
    the batch, either from sequential enumeration or from the given labels.
    """
    ballot_manifest_dict = OrderedDict()
    for i in manifest:
        # assert that the entry is a string with a comma in it
        # pull out batch label
        (batch, val) = i.split(",")
        batch = batch.strip()
        val = val.strip()
        if batch in ballot_manifest_dict.keys():
            raise ValueError('batch is listed more than once')
        else:
            ballot_manifest_dict[batch] = []

        # parse what comes after the batch label
        if '(' in val:     # list of identifiers
            # TO DO: use regex to remove )(
            val = val[1:-1] # strip out the parentheses
            ballot_manifest_dict[batch] += [int(num) for num in val.split()]
        elif ':' in val:   # range of identifiers
            limits = val.split(':')
            ballot_manifest_dict[batch] += list(range(int(limits[0]), \
                                             int(limits[1])+1))
        else:  # this should be an integer number of ballots
            try:
                ballot_manifest_dict[batch] += list(range(1, int(val)+1))
            except:
                print('malformed row in ballot manifest:\n\t', i)
    return ballot_manifest_dict


def unique_manifest(parsed_manifest):
    """
    Create a single ballot manifest with unique IDs for each ballot.
    Identifiers are unique across batches, so the ballots can be considered
    in a canonical order.
    """
    second_manifest = {}
    ballots_counted = 0
    for batch in parsed_manifest.keys():
        batch_size = len(parsed_manifest[batch])
        second_manifest[batch] = list(range(ballots_counted + 1, \
                                    ballots_counted + batch_size + 1))
        ballots_counted += batch_size
    return second_manifest


def find_ballot(ballot_num, unique_ballot_manifest):
    """
    Find ballot among all the batches

    Input
    -----
    ballot_num : int
        a ballot number that was sampled
    unique_ballot_manifest : dict
        ballot manifest with unique IDs across batches


    Returns
    -------
    tuple : (original_ballot_label, batch_label, which_ballot_in_batch)
    """
    for batch, ballots in unique_ballot_manifest.items():
        if ballot_num in ballots:
            position = ballots.index(ballot_num) + 1
            return (batch, position)
    print("Ballot %i not found" % ballot_num)
    return None


def sample_from_manifest(filename, sample, stratum_size):
    """
    Sample from the ballot manifest
    
    
    """
    ballot_manifest = read_manifest_from_csv(filename)
    manifest_parsed = parse_manifest(ballot_manifest)
    listed = np.sum([len(v) for v in manifest_parsed.values()])
    if listed != stratum_size:
        print("WARNING: the number of ballots in the ballot manifest is ",\
              listed, "but total number of reported votes is", stratum_size)
    
    unique_ballot_manifest = unique_manifest(manifest_parsed)

    ballots_sampled = []
    m = np.zeros_like(sample, dtype=bool)
    m[np.unique(sample, return_index=True)[1]] = True
    for s in sample[m]:
        batch_label, which_ballot = find_ballot(s, unique_ballot_manifest)
        if s in sample[~m]:
            ballots_sampled.append([s, batch_label, which_ballot, np.sum(np.array(sample) == s)])
        else:
            ballots_sampled.append([s, batch_label, which_ballot, 1])
        
    ballots_sampled.sort(key=lambda x: x[2]) # Sort second on order within batches
    ballots_sampled.sort(key=lambda x: x[1]) # Sort first based on batch label
    ballots_sampled.insert(0,["sampled ballot", "batch label", "which ballot in batch", "# times sampled"])
    return ballots_sampled



################################################################################
############################## Do the audit! ###################################
################################################################################

def audit_contest(candidates, winners, losers, stratum_sizes,\
                  n1, n2, o1_obs, o2_obs, u1_obs, u2_obs, observed_poll, \
                  risk_limit, gamma, stepsize):
    """
    Use SUITE to calculate risk of each (winner, loser) pair
    given the observed samples in the CVR and no-CVR strata.

    Parameters
    ----------
    candidates : dict
        OrderedDict with candidate names as keys and 
        [CVR votes, no-CVR votes, total votes] as values
    winners : list
        names of winners
    losers : list
        names of losers
    stratum_sizes : list
        list with total number of votes in the CVR and no-CVR strata
    n1 : int
        size of sample already drawn in the ballot comparison stratum
    n2 : int
        size of sample already drawn in the ballot polling stratum
    o1_obs : int
        observed number of ballots with 1-vote overstatements in the CVR stratum
    o2_obs : int
        observed number of ballots with 2-vote overstatements in the CVR stratum
    u1_obs : int
        observed number of ballots with 1-vote understatements in the CVR
        stratum
    u2_obs : int
        observed number of ballots with 2-vote understatements in the CVR
        stratum
    observed_poll : dict
        Dict with candidate names as keys and number of votes in the no-CVR
        stratum sample as values
    risk_limit : float
        risk limit
    gamma : float
        gamma from Lindeman and Stark (2012)
    stepsize : float
        stepsize for the discrete bounds on Fisher's combining function
    Returns
    -------
    dict : attained risk for each (winner, loser) pair in the contest
    """
    audit_pvalues = {}

    for k in product(winners, losers):
        N_w1 = candidates[k[0]][0]
        N_w2 = candidates[k[0]][1]
        N_l1 = candidates[k[1]][0]
        N_l2 = candidates[k[1]][1]
        n2w = observed_poll[k[0]]
        n2l = observed_poll[k[1]]
        reported_margin = (N_w1+N_w2)-(N_l1+N_l2)
        
        if stratum_sizes[1] == 0:
            audit_pvalues[k] = ballot_comparison_pvalue(n=n1, \
                        gamma=gamma, \
                        o1=o1_obs, u1=u1_obs, o2=o2_obs, u2=u2_obs, \
                        reported_margin=reported_margin, \
                        N=stratum_sizes[0], \
                        null_lambda=1)
        
        elif stratum_sizes[0] == 0:
            sam = np.array([0]*n2l+[1]*n2w+[np.nan]*(n2-n2w-n2l))
            audit_pvalues[k] = ballot_polling_sprt(\
                                sample=sam, \
                                popsize=stratum_sizes[1], \
                                alpha=risk_limit, \
                                Vw=N_w2, Vl=N_l2, \
                                null_margin=0)['pvalue']
        else:
            if n1 == 0:
                cvr_pvalue = lambda alloc: 1
            else:
                cvr_pvalue = lambda alloc: ballot_comparison_pvalue(n=n1, \
                            gamma=gamma, \
                            o1=o1_obs, u1=u1_obs, o2=o2_obs, u2=u2_obs, \
                            reported_margin=reported_margin, \
                            N=stratum_sizes[0], \
                            null_lambda=alloc)

            if n2 == 0:
                nocvr_pvalue = lambda alloc: 1
            else:
                sam = np.array([0]*n2l+[1]*n2w+[np.nan]*(n2-n2w-n2l))
                nocvr_pvalue = lambda alloc: ballot_polling_sprt(\
                                    sample=sam, \
                                    popsize=stratum_sizes[1], \
                                    alpha=risk_limit, \
                                    Vw=N_w2, Vl=N_l2, \
                                    null_margin=(N_w2-N_l2) - \
                                      alloc*reported_margin)['pvalue']
            bounding_fun = create_modulus(n1=n1, n2=n2, \
                                          n_w2=n2w, \
                                          n_l2=n2l, \
                                          N1=stratum_sizes[0], \
                                          V_wl=reported_margin, gamma=gamma)
            res = maximize_fisher_combined_pvalue(N_w1=N_w1, N_l1=N_l1,\
                             N1=stratum_sizes[0], \
                             N_w2=N_w2, N_l2=N_l2, \
                             N2=stratum_sizes[1], \
                             pvalue_funs=(cvr_pvalue, nocvr_pvalue), \
                             stepsize=stepsize, \
                             modulus=bounding_fun, \
                             alpha=risk_limit)
            audit_pvalues[k] = res['max_pvalue']

    return audit_pvalues


################################################################################
############################## Unit testing ####################################
################################################################################

def test_initial_n():
    """
    Assume N1 = N2 = 500, n1 = n2 \equiv n,
    and the margins V1 = V2 = V/2 are identical in each stratum.
    w got 60% of the vote and l got 40%.
    It's known that there are no invalid ballots or votes for other candidates.
    Assume there are no errors in the comparison stratum and the sample
    proportions in the polling stratum are 60% and 40%.

    In the polling stratum,
        $$c(\lambda) = V/2 - (1-\lambda)V = 200\lambda - 100$$
    and
        $$N_w(\lambda) = (N2 + c(\lambda))/2 = 200 + 100\lambda.$$
    Therefore the Fisher combination function is
        $$ \chi(\lambda) = -2\[ \sum_{i=1}^{np_w - 1} \log(200+100\lambda - i)
            - \sum_{i=1}^{np_w - 1} \log(300 - i) +
            \sum_{i=1}^{np_\ell - 1} \log(300-100\lambda - i)
            - \sum_{i=1}^{np_\ell - 1} \log(200 - i)
            + n\log( 1 - \frac{\lambda}{5\gamma} )\] $$
    """

    chi_5percent = scipy.stats.chi2.ppf(1-0.05, df=4)
    chi_10percent = scipy.stats.chi2.ppf(1-0.10, df=4)

    # sample sizes: n = 50 in each stratum. Not sufficient.
    chi50 = lambda lam: -2*( np.sum(np.log(200 + 100*lam - np.arange(30))) - \
        np.sum(np.log(300 - np.arange(30))) + np.sum(np.log(300 - 100*lam - \
        np.arange(20))) - np.sum(np.log(200 - np.arange(20))) + \
        50*np.log(1 - lam/(5*gamma)))

    # Valid lambda range is (-2, 3)
    approx_chisq_min = np.nanmin(list(map(chi50, np.arange(-2,3,0.05))))
    np.testing.assert_array_less(approx_chisq_min, chi_5percent)

    # sample sizes: n = 70 in each stratum. Sufficient.
    chi70 = lambda lam: -2*( np.sum(np.log(200 + 100*lam - np.arange(42))) - \
        np.sum(np.log(300 - np.arange(42))) + np.sum(np.log(300 - 100*lam - \
        np.arange(28))) - np.sum(np.log(200 - np.arange(28))) + \
        70*np.log(1 - lam/(5*gamma)))
    approx_chisq_min = np.nanmin(list(map(chi70, np.arange(-2,3,0.05))))
    np.testing.assert_array_less(chi_10percent, approx_chisq_min)
    np.testing.assert_array_less(chi_5percent, approx_chisq_min)

    n = estimate_n(N_w1 = 300, N_w2 = 300, N_l1 = 200, N_l2 = 200,\
           N1 = 500, N2 = 500, o1_rate = 0, o2_rate = 0,\
           u1_rate = 0, u2_rate = 0, n_ratio = 0.5,
           risk_limit = 0.05, gamma = 1.03905)
    np.testing.assert_equal(n[0] <= 70 and n[0] > 30, True)
    if (n[0]+n[1]) % 2 == 1:
        np.testing.assert_equal(n[0], n[1]+1)
    else:
        np.testing.assert_equal(n[0], n[1])
    
    # sample sizes: n = 55 in each stratum. Should be sufficient.
    chi55 = lambda lam: -2*( np.sum(np.log(200 + 100*lam - np.arange(33))) - \
        np.sum(np.log(300 - np.arange(33))) + np.sum(np.log(300 - 100*lam - \
        np.arange(22))) - np.sum(np.log(200 - np.arange(22))) + \
        55*np.log(1 - lam/(5*gamma)))
    approx_chisq_min = np.nanmin(list(map(chi55, np.arange(-2,3,0.05))))
    np.testing.assert_array_less(chi_10percent, approx_chisq_min)
    np.testing.assert_array_less(chi_5percent, approx_chisq_min)


if __name__ == "__main__":
    test_initial_n()
