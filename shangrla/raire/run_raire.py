from shangrla.raire.raire_utils import *
from shangrla.raire.raire import compute_raire_assertions
from shangrla.raire.sample_estimator import *

import numpy as np

import argparse

def main():
  """
  The RAIRE CLI
  """
  parser = argparse.ArgumentParser()
  parser.add_argument('-i', dest='input', required=True)
  parser.add_argument('-bp', dest='bp', action='store_true')
  parser.add_argument('-v', dest='verbose', action='store_true')

  parser.add_argument('-agap', dest='agap', type=float, default=0)

  # Used for estimating sample size for assertions if desired.
  parser.add_argument('-r', dest='rlimit', type=float, default=0.10)

  # Used when estimating sample size given non zero error rate for comparison
  # audits. No sample size estimator in sample_estimator.py for ballot polling
  # with non-zero error rate.
  parser.add_argument('-e1', dest='erate1', type=float, default=0.002)
  parser.add_argument('-e2', dest='erate2', type=float, default=0)
  parser.add_argument('-seed', dest='seed', type=int, default=1234567)
  parser.add_argument('-reps', dest='reps', type=int, default=100)

  args = parser.parse_args()


  contests, cvrs = load_contests_from_raire(args.input)

  est_fn = bp_estimate if args.bp else cp_estimate

  np.seterr(all="ignore")

  for contest in contests:
      audit = compute_raire_assertions(contest, cvrs, contest.winner, 
          est_fn, args.verbose, agap=args.agap)

      N = contest.tot_ballots

      max_est = 0

      if audit == []:
          print(f"File {args.input}, Contest {contest.name}, No audit possible")
      else:

          for asrt in audit:
              est = None
              tally_other = N - asrt.votes_for_winner - asrt.votes_for_loser
              amean = (asrt.votes_for_winner + 0.5*tally_other)/N
              est = sample_size(amean, asrt.votes_for_winner, \
                  asrt.votes_for_loser, tally_other, args, N, polling=args.bp)

              est = min(est, N) # Cut off at a full recount

              max_est = max(max_est, est)

              est_p = 100*(est/N)
              if args.verbose:
                  print("{}, est {},{}%".format(asrt.to_str(), est, est_p))

      if max_est != 0:
          max_est = min(max_est, N)
          max_est_p = 100*(max_est/N)
          print(f"File {args.input}, Contest {contest.name}, asn {max_est}, {max_est_p:.2f}%")
