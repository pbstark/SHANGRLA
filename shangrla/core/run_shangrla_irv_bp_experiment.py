import json
import numpy as np
import csv
import argparse
import os
from pathlib import Path

from shangrla.core.Audit      import Audit, Assertion, Contest, CVR
from shangrla.core.NonnegMean import NonnegMean


def main():
    """
    Run SHANGRLA for ballot-polling audits of an IRV contest.

    The `risk_limit` option (AFAIK) has no effect because this script will
    output the full p-value history(ies).
    """
    # -------------------------------------------------------------------------
    # Parse arguments.

    parser = argparse.ArgumentParser(
            description="Runs ballot-polling audits for IRV contests.")
    parser.add_argument("CVRS",
            help="file with ballots in RAIRE format")
    parser.add_argument("ASSERTIONS",
            help="JSON file with RAIRE assertions")
    parser.add_argument("WINNER",
            help="name of the reported winning candidate")
    parser.add_argument("ORDERINGS",
            help="CSV file with orderings of the ballots")
    parser.add_argument("OUTFILE",
            help="CSV output file with the calculated p-values")
    parser.add_argument("-l", "--risk_limit",
            help="risk limit (default: 0.05)",
            type=int,
            default=0.05)
    args = parser.parse_args()


    # -------------------------------------------------------------------------
    # Load data.

    # Load CVR data file.
    cvr_input = []
    with open(args.CVRS) as f:
        cvr_reader = csv.reader(f, delimiter=',', quotechar='"')
        for row in cvr_reader:
            cvr_input.append(row)
    print(f'Read {len(cvr_input)} rows from: {args.CVRS}')

    # Load the assertions for the IRV contest.
    with open(args.ASSERTIONS) as f:
        assertion_json = json.load(f)['audits'][0]['assertions']
    print(f'Read {len(assertion_json)} assertions from: {args.ASSERTIONS}')

    # Load the orderings file.
    orderings = np.loadtxt(args.ORDERINGS, int, delimiter=',', skiprows=1)
    print(f'Read {orderings.shape[0]} orderings from: {args.ORDERINGS}')


    # -------------------------------------------------------------------------
    # Munge data.

    # Import CVRs.
    cvr_list, num_cvrs_read = CVR.from_raire(cvr_input)
    max_cards = len(cvr_list)
    print(f'After merging, there are CVRs for {max_cards} cards')

    # Extract the set of candidates from loaded data.
    #
    # This is from the 2nd line of input file, 4th element onwards.
    candidates = cvr_input[1][3:]

    # Set specifications for the audit.
    audit = Audit.from_dict({
        'strata' : {'stratum1' : {'use_style'   : True,
                                  'replacement' : True}
            }
        })
    contest_dict = {'1':{'name'             : 'Contest1',
                         'risk_limit'       : args.risk_limit,
                         'cards'            : max_cards,
                         'choice_function'  : Contest.SOCIAL_CHOICE_FUNCTION.IRV,
                         'n_winners'        : 1,
                         'candidates'       : candidates,
                         'winner'           : [args.WINNER],
                         'assertion_file'   : args.ASSERTIONS,
                         'assertion_json'   : assertion_json,
                         'audit_type'       : Audit.AUDIT_TYPE.POLLING,
                         'test'             : NonnegMean.alpha_mart,
                         'estim'            : NonnegMean.shrink_trunc
                         }
                    }
    contests = Contest.from_dict_of_dicts(contest_dict)

    # Construct the dict of dicts of assertions for each contest.
    Assertion.make_all_assertions(contests)
    audit.check_audit_parameters(contests)

    # Calculate margins for each assertion.
    min_margin = Assertion.set_all_margins_from_cvrs(audit, contests, cvr_list)

    # Some output.
    print(f"Smallest margin in contest: {min_margin}")
    print("Assorters and margins:")
    Contest.print_margins(contests)

    # Calculate all of the p-values.
    pvalue_histories_array = calc_pvalues_all_orderings(contests, cvr_input, orderings)

    # Create output directory (it not already present).
    Path(os.path.dirname(args.OUTFILE)).mkdir(parents=True, exist_ok=True)

    np.savetxt(args.OUTFILE, pvalue_histories_array, fmt='%.3f', delimiter=',')
    print(f"Full p-value histories saved to: {args.OUTFILE}")


# =============================================================================
# Define functions.

# Shuffle raw RAIRE-formatted CVRs according to a given ordering.
#
# Need to concatenate [0, 1] at the start, to avoid shuffling the file headers.
# The other indices need to be translated accordingly (by +1).
def shuffle(cvrs, ordering):
    neworder = np.concatenate(([0, 1], ordering + 1))
    cvrs_shuffled = [cvrs[i] for i in neworder]
    return(cvrs_shuffled)

# Extract a 'merged' p-value history, combined across all assertions.
#
# This returns a single list of p-values, which is the p-value at each stage of
# sampling.  That is, the smallest value of `alpha` for which the audit may
# terminate (and certify) at that stage.
def merge_pvalues(assertions_dict):
    pvalue_histories = []
    for asrtn in assertions_dict:
        a = assertions_dict[asrtn]
        phist = a.p_history
        phist_running_min = np.minimum.accumulate(phist)
        pvalue_histories.append(phist_running_min)
    pvalue_histories_stacked = np.stack(pvalue_histories)
    pvalue_histories_merged  = np.amax(pvalue_histories_stacked, axis=0)
    return(pvalue_histories_merged)

# Calculate p-values for a given ordering.
def calc_pvalues_single_ordering(contests, cvr_input, orderings, ordering_index):
    #print("Working on ordering {}".format(ordering_index))

    # Shuffle ballots according to the given ordering.
    cvr_input_shuffled = shuffle(cvr_input, orderings[ordering_index, :])

    # Import shuffled CVRs.
    shuffled_ballots, _ = CVR.from_raire(cvr_input_shuffled)

    # Find measured risks for all assertions.
    Assertion.set_p_values(contests, shuffled_ballots, None)

    # Extract all of the p-value histories and combine them.
    pvalues = merge_pvalues(contests['1'].assertions)

    return(pvalues)

# Calculate p-values for a set of orderings.
def calc_pvalues_all_orderings(contests, cvr_input, orderings):
    n_orderings = orderings.shape[0]
    pvalue_list = []
    for o in range(n_orderings):
        pvalue_list.append(calc_pvalues_single_ordering(contests, cvr_input, orderings, o))
    pvalue_array = np.stack(pvalue_list, axis=-1)
    return(pvalue_array)

# =============================================================================
# name-main idiom
#
# Runs main() if invoked as a script.

if __name__ == "__main__":
    main()
