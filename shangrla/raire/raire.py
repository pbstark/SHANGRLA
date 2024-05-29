# Copyright (C) 2022 Michelle Blom
#
# This program is free software: you can redistribute it and/or modify
# it under the terms of the GNU Affero General Public License as published
# by the Free Software Foundation, either version 3 of the License, or
# (at your option) any later version.
#
# This program is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
# GNU Affero General Public License for more details.

# You should have received a copy of the GNU Affero General Public License
# along with this program.  If not, see <https://www.gnu.org/licenses/>.


from shangrla.raire.raire_utils import (
  NEBAssertion,
  RaireFrontier,
  RaireNode,
  find_best_audit,
  perform_dive,
  manage_node
)

import numpy as np
import sys


def compute_raire_assertions(
    contest, cvrs, winner, asn_func, log, stream=sys.stdout, agap=0,\
    seed=123456
):

    """

    Inputs:
        contest        - the contest being audited (Contest structure)

        cvrs           - mapping of ballot_id to votes:
                {
                    'ballot_id': {
                        'contest': {
                            'candidate1': 1,
                            'candidate2': 0,
                            'candidate3': 2,
                            'candidate4': 3,
                            ...
                        }
                    ...
                }

        winner         - reported winner of the contest

        asn_func       - function that takes three values as input: tally for 
                         the winner of an assertion; the loser; and the total 
                         number of auditable ballots. Returns an estimate of 
                         how difficult a RAIRE assertion with that margin will
                         be to audit.

        log            - flag indicating if logging statements should
                         be printed during the algorithm.

        stream         - stream to which logging statements should
                         be printed.
        
        agap           - allowed gap between the lower and upper bound
                         on expected audit difficulty. Once these bounds
                         converge (to within 'agap') algorithm can stop
                         and return  audit configuration found. Generally,
                         keep this at 0 unless the algorithm is not 
                         terminating in a reasonable time. Then set it to
                         as small a value as possible, and increase, until
                         the algorithm terminates. For some instances, the
                         difference between the lower and upper bound on 
                         expected audit difficulty gets to a point where it
                         is quite small, but doesn't converge. 

    Outputs:
        A list of RaireAssertions to be audited. If this collection of
        assertions is found to hold, then all alternate outcomes, in which
        an alternate candidate to 'winner' wins, can be ruled out. 
    """

    ncands = len(contest.candidates)
    
    # First look at all of the NEB assertions that could be formed for
    # this contest. We will refer to this matrix when examining the best
    # way to prune branches of the "alternate outcome space". 
    nebs = {c : { d : None for d in contest.candidates} 
        for c in contest.candidates} 

    for c in contest.candidates:
        for d in contest.candidates:
            if c == d: 
                continue

            asrn = NEBAssertion(contest.name, c, d)
            
            tally_c = 0
            tally_d = 0
            for _,r in cvrs.items():
                tally_c += asrn.is_vote_for_winner(r)
                tally_d += asrn.is_vote_for_loser(r)

            if tally_c > tally_d:
                asrn.difficulty = asn_func(tally_c, tally_d, \
                    contest.tot_ballots - (tally_c + tally_d), \
                    contest.tot_ballots)

                asrn.votes_for_winner = tally_c
                asrn.votes_for_loser = tally_d

                nebs[c][d] = asrn


    # The RAIRE algorithm progressively searches through the space of 
    # alternate election outcomes, viewing this space as a tree. We store
    # the current leaves of this tree, at any point in the search, in a 
    # list called 'frontier'. Each leaf is a (potentially) partial election
    # outcome, describing the tail of the elimination sequence and eventual
    # winner. All candidates not mentioned in this tail are assumed to have
    # already been eliminated. 

    ballots = [blt[contest.name] for _,blt in cvrs.items() 
        if contest.name in blt]

    # This is a running lowerbound on the overall difficulty of the 
    # election audit. 
    lowerbound = -10

    # Construct initial frontier. 
    frontier = RaireFrontier()

    # Our frontier initially has a node for each alternate election outcome
    # tail of size two. The last candidate in the tail is the ultimate winner. 
    for c in contest.candidates:
        if c == winner: continue

        for d in contest.candidates:
            if c == d: continue

            newn = RaireNode([d,c])
            newn.expandable = True if ncands > 2 else False

            find_best_audit(contest, ballots, nebs, newn, asn_func)

            if log:
                print("TESTED ", file=stream, end='')
                newn.display(stream=stream)
                if newn.best_assertion != None:
                    print("   Best audit ", file=stream, end='')
                    newn.best_assertion.display(stream=stream)

            frontier.insert_node(newn)

    # Flag to keep track of whether a full manual recount will be required
    audit_not_possible = False

    if log:
        print("===============================================", file=stream)
        print("Initial Frontier", file=stream)
        frontier.display(stream=stream)
        print("===============================================", file=stream)
    

    # -------------------- Find Assertions -----------------------------------
    while not audit_not_possible:
        # Check whether we can stop searching for assertions.
        max_on_frontier = max([node.estimate for node in frontier.nodes])

        if agap > 0 and lowerbound > 0 and max_on_frontier-lowerbound <= agap:
            # We can rule out all branches of the tree with assertions that
            # have a difficulty that is <= lowerbound. 
            break

        to_expand = frontier.nodes[0]

        # We can also stop searching if all nodes on our frontier are leaves.
        if not to_expand.expandable:
            break

        frontier.nodes.pop(0)

        if to_expand.best_ancestor != None and \
            to_expand.best_ancestor.estimate <= lowerbound:
            frontier.replace_descendents(to_expand.best_ancestor, log,
                stream=stream)

            continue

        if to_expand.estimate <= lowerbound:
            to_expand.expandable = False
            frontier.insert_node(to_expand)
            continue

        #--------------------------------------------------------------------
        # "Dive" straight from "to_expand" down to a leaf -- one of its
        # decendants -- and find the least cost assertion to rule out the
        # branch of the alternate outcomes tree that ends in that leaf. We
        # know that this assertion will be part of the audit, as we have
        # to rule out all branches. 
        if not to_expand.dive_node:
            dive_lb = perform_dive(to_expand, contest, ballots, nebs, \
                asn_func, lowerbound, frontier, log, stream=stream)

            if dive_lb == np.inf:
                # The particular branch we dived along cannot be ruled out
                # with an assertion.
                audit_not_possible = True
                if log:
                    print("Diving finds that audit is not possible",
                        file=stream)
                break

            if log:
                print("Diving LB {}, Current LB {}".format(dive_lb, 
                    lowerbound), file=stream)

            # We can use our new knowledge of the "best" way to rule out
            # the branch to update our "lowerbound" on the overall "difficulty"
            # of the eventual audit.
            lowerbound = max(lowerbound, dive_lb)

            if to_expand.best_ancestor != None and \
                to_expand.best_ancestor.estimate <= lowerbound:
                frontier.replace_descendents(to_expand.best_ancestor, log,
                    stream=stream)

                continue

            if to_expand.estimate <= lowerbound:
                to_expand.expandable = False
                frontier.insert_node(to_expand)
                continue

        #--------------------------------------------------------------------

        if log:
            print("  Expanding node ", file=stream, end='')
            to_expand.display(stream=stream)

        # Find children of current node, and find the best assertions that 
        # could be used to prune those nodes from the tree of alternate
        # outcomes.
        for c in contest.candidates:
            if not c in to_expand.tail and not c in to_expand.explored:
                newn = RaireNode([c] + to_expand.tail)
                newn.expandable = False if len(newn.tail) == ncands else True

                # Assign a 'best ancestor' to the new node. 
                newn.best_ancestor = to_expand.best_ancestor if \
                    to_expand.best_ancestor != None and \
                    to_expand.best_ancestor.estimate <= to_expand.estimate \
                    else to_expand

                find_best_audit(contest, ballots, nebs, newn, asn_func)

                if log:
                    print("TESTED ", file=stream, end='')
                    newn.display(stream=stream)

                audit_not_possible, lowerbound, _ = manage_node(newn,frontier,\
                    lowerbound, log, stream=stream)


            if audit_not_possible: break    

        
        if log:
            print("Size of frontier {}, current lower bound {}".format(
                len(frontier.nodes), lowerbound))

        if audit_not_possible: break 

    # If a full recount is required, return empty list.
    if audit_not_possible: 
        if log:
            print("AUDIT NOT POSSIBLE", file=stream)

        return []

    # ------------------------------------------------------------------------
    # Some assertions will be used to rule out multiple branches of our
    # alternate outcome tree. Form a list of all these assertions, without
    # duplicates.
    assertions = []

    for node in frontier.nodes:
        skip = False
        for assrtn in assertions:
            if node.best_assertion.same_as(assrtn):
                assrtn.rules_out.update(node.best_assertion.rules_out)
                skip = True
                break

        if not skip:
            assertions.append(node.best_assertion)

    # Assertions will be sorted in order of how much of the alternate
    # outcome space they rule out (most to least).
    sorted_assertions = sorted(assertions)    
    len_assertions = len(sorted_assertions)

    final_audit = []

    if sorted_assertions != []:
        final_audit = [sorted_assertions.pop(0)]

        for assertion in sorted_assertions:
            subsumed = False
            for fasrtn in final_audit:
                if fasrtn.subsumes(assertion):
                    fasrtn.rules_out.update(assertion.rules_out) 
                    if log:
                        print("{} SUBSUMES {}".format(fasrtn.to_str(),
                            assertion.to_str()), file=stream)

                    subsumed = True
                    break

            if not subsumed:
                final_audit.append(assertion)


    if log:
        print("===============================================", file=stream)
        print("ASSERTIONS:", file=stream)
        for assertion in final_audit:
            assertion.display(stream=stream)
        print("===============================================", file=stream)
        
    return final_audit  


