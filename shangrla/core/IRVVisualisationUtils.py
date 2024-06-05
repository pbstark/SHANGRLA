import svgling
from svgling.figure import Caption, RowByRow
from colorama import Fore
import functools
from warnings import warn
from collections import namedtuple

LeafNode = namedtuple("LeafNode", "cand, NEBTagList, IRVTagList")


# Convert a tree in list form into the same tree in tuple form suitable for
# svgling.
def treeListToTuple(t):
    # If t is an empty list, we shouldn't have got this far.
    if not t:
        warn("Error: empty list in tree drawing")
    # Leaf.  Return the name of the candidate and the assertions we've excluded it with.

    tag = ""

    if len(t) == 1:
        # this is a node to be pruned, with two lists of (number, bool) tuples indicating
        # the assertion numbers that justify pruning it, and whether each has been proved.
        # We find the logical OR of all the 'proven' booleans - if any are confirmed, this
        # prune is confirmed.
        node = t[0]
        if node.NEBTagList:
            tag += (
                "NEB "
                + ",".join(str(n[0]) for n in node[1])
                + "\n"
                + buildConfTag(node[1])
            )
        if node.NEBTagList and node.IRVTagList:
            tag += "\n"
        if node.IRVTagList:
            tag += (
                "IRV "
                + ",".join(str(n[0]) for n in node[2])
                + "\n"
                + buildConfTag(node[2])
            )
        if not (node.NEBTagList or node.IRVTagList):
            tag = "***Unpruned leaf. RAIRE assertions do not exclude all other winners!***"
        return (node[0], tag)

    # Otherwise recurse.
    else:
        tList = []
        for branch in t[1]:
            tList.append(treeListToTuple(branch))
        return (t[0],) + tuple(tList)


# Expects a list of (_, boolean) tuples.
# Takes the logical OR and returns either "Confirmed" if True
# or "Unconfirmed" if False.
def buildConfTag(numBoolList):
    if len(numBoolList) == 0:
        return "Error: no truth values for this assertion."
    truthval = (functools.reduce(lambda a, b: (None, a[1] or b[1]), numBoolList))[1]
    if truthval:
        return "Confirmed"
    else:
        return "Unconfirmed"


# Parses a json file containing assertions.
# Complicated by the fact that we have two slightly different
# formats, one produced by RAIRE and the other as a log file
# of the audit process.  Some of the field names are slightly
# different.
def parseAssertions(auditfile, candidatefile, contest_id=None):

    RLALogfile = False
    auditsArray = []

    # For audit-generated log format, the first is selected by default (i.e. first in
    # sorted list of IDs - which assumes IDs are string-representations of integers).
    # The optional contest_id can be passed to select a different contest
    #
    # FIXME: for the RAIRE log format, it's hardcoded to draw only the first audit for now,
    # though it parses all of them.
    # 'first' isn't even well-defined for a DICT so this needs to be fixed.
    if "Audit" in auditfile and "seed" in auditfile["Audit"]:
        # Assume this is formatted like a log file from assertion-RLA.
        RLALogfile = True

        # Get sorted list of contest IDs (converted back into str format)
        contestNumList = sorted(list(map(int, auditfile["contests"])))
        contestNumList = list(map(str, contestNumList))

        auditsDict = {}
        for contestNum in auditfile["contests"]:
            contest = auditfile["contests"][contestNum]

            auditsDict[contestNum] = contest
            if contest["choice_function"] != "IRV":
                warn("IRV Visualisations: visualising a non-IRV assertion set.")

            if contest["n_winners"] != 1:
                warn("IRV contest with either zero or >1 winner")

        try:
            audit = auditsDict[str(contest_id)]
        except KeyError:
            audit = auditsDict[contestNumList[0]]
        apparentWinner = audit["winner"][0]
        print("apparentWinner = " + apparentWinner)
        print("candidates = " + str(audit["candidates"]))
        apparentNonWinners = audit["candidates"].copy()
        apparentNonWinners.remove(apparentWinner)
        #apparentNonWinners = audit["candidates"].remove(apparentWinner)
        print("apparent Non Winners: " + str(apparentNonWinners))

        # SHANGRLA IRV contest audits write an "assertion_json" section to the log.json file,
        # while plurality audits do not
        assertions = audit["assertions"]
        try:
            assertion_json = audit["assertion_json"]
        except KeyError:
            assertion_json = []

    else:
        # Assume this is formatted like the assertions output from RAIRE
        auditsArray = auditfile["audits"]

        audit = auditsArray[0]
        apparentWinner = audit["winner"]
        apparentNonWinners = audit["eliminated"]
        assertions = audit["assertions"]
        assertion_json = []

    apparentWinnerName = findCandidateName(apparentWinner, candidatefile)
    print("Apparent winner: " + "\n" + printTuple((apparentWinner, apparentWinnerName)))

    apparentNonWinnersWithNames = findListCandidateNames(
        apparentNonWinners, candidatefile
    )
    print("Apparently eliminated:")
    print(",\n".join(list(map(printTuple, apparentNonWinnersWithNames))))
    print("\n")

    # WOLosers is a list of tuples - the first element of the tuple is the loser,
    # the second element is the set of all the candidates it loses relative to.
    WOLosers = []

    # IRVElims is also a list of tuples - the first element is the candidate,
    # the second is the set of candidates already eliminated.
    # An IRVElim assertion states that the candidate can't be the next
    # eliminated when the already-eliminated candidates are exactly the set
    # in the second element of the tuple.
    IRVElims = []

    for index, a in enumerate(assertions.values()):

        # Extract the "assertion_json" dict, if present
        try:
            a_detail = assertion_json[index]
        except IndexError:
            a_detail = {}

        if RLALogfile:
            if "proved" in a:
                proved = a["proved"]
            else:
                warn("No proved information in log file - assuming all unconfirmed.")
                proved = False
        else:
            if ("proved" in a) and (a["proved"] == "True"):
                proved = True
            else:
                proved = False

        # Pull winner/loser info from the "assertion_json" dict if available
        if "assertion_type" in a_detail.keys():
            if a_detail["assertion_type"] == "WINNER_ONLY":
                if a_detail["already_eliminated"] != "":
                    # VT: Not clear whether we should go on or quit at this point.
                    warn("Error: Not-Eliminated-Before assertion with nonempty already_eliminated list.")

                l = a_detail["loser"]
                w = a_detail["winner"]
                WOLosers.append((l,w,proved))

            elif a_detail["assertion_type"] == "IRV_ELIMINATION":
                l = a_detail["winner"]
                IRVElims.append((l,set(a_detail["already_eliminated"]),proved))

        # Otherwise, pull it from the "assertion" dict (this assumes non-IRV)
        else:
            l = a["loser"]
            w = a["winner"]
            WOLosers.append((l,w,proved))

    return (
        (apparentWinner, apparentWinnerName),
        apparentNonWinnersWithNames,
        WOLosers,
        IRVElims,
    )

def printTuple(t):
    return str(t[0]) + "-" + str(t[1])


# Given a candidate ID, find their name in the Candidate Manifest.
def findCandidateName(ID, candidateFile):
    candidates = candidateFile["List"]
    for c in candidates:
        if str(c["Id"]) == ID:
            return c["Description"]

    return ""


# Given a list of candidate IDs, build a list of (ID, Name) tuples.
def findListCandidateNames(IDList, candidateFile):
    return list(map(lambda id: (id, findCandidateName(id, candidateFile)), IDList))


# This takes a root candidate c and a set S of candidates still to
# be built in to the tree (i.e. those to be eliminated earlier, closer to the leaves)
# it checks whether any assertions apply to this point in the elimination and, if so,
# prunes the tree here.
def buildRemainingTreeAsLists(c, S, WOLosers, IRVElims):

    # If c is in the list of candidates yet to be eliminated, this is a bug.
    if c in S:
        print("Error: c is in S.  c = " + str(c) + ". S = " + str(S) + ".\n")

    pruneThisBranch = False
    NEBTags = []
    IRVTags = []

    # if c is a loser defeated by a candidate in S, prune here.
    # Tag with NEB assertion number we used to prune.
    for loser in WOLosers:
        if c == loser[0] and ((loser[1] in S)):
            pruneThisBranch = True
            NEBTags.append((WOLosers.index(loser), loser[2]))

    # if c cannot be eliminated by IRV in exactly the case where S is the already-eliminated set,
    # prune here.  Tag with IRV assertion number we used to prune.
    for winner in IRVElims:
        if c == winner[0] and winner[1] == S:
            pruneThisBranch = True
            IRVTags.append((IRVElims.index(winner), winner[2]))

    if pruneThisBranch:
        # Base case: if we prune here, tag it with all the assertions
        # that could be used to prune.
        tree = [LeafNode(cand=c, NEBTagList=NEBTags, IRVTagList=IRVTags)]
    # if S is empty, return the leaf
    # Note that this indicates an error in the RAIRE audit
    # process - we're producing a tree
    # in which we haven't pruned all the leaves.
    # The intention is to make it visually obvious to an auditor.
    # Hence the ***
    elif not S:
        return [LeafNode(cand=c, NEBTagList=[], IRVTagList=[])]
        warn(
            "***Unpruned leaf "
            + c
            + ". RAIRE assertions do not exclude all other winners!***"
        )
    else:
        # if we didn't prune here, recurse
        tree = [c, []]
        for c2 in S:
            smallerSet = S.copy()
            smallerSet.remove(c2)
            tree[1].append(
                buildRemainingTreeAsLists(c2, smallerSet, WOLosers, IRVElims)
            )

    return tree


def printAssertions(WOLosers, IRVElims):
    if len(WOLosers) != 0:
        print("Not-Eliminated-Before assertions: ")
    for loser in WOLosers:
        proofString = makeProofString(loser, Fore.RED)
        print(
            proofString
            + "NEB {0:2d}: ".format(WOLosers.index(loser))
            + Fore.BLACK
            + "Candidate "
            + str(loser[1])
            + " cannot be eliminated before "
            + str(loser[0])
            + "."
        )

    if len(IRVElims) != 0:
        print("\n")
        print("IRV assertions: ")
    for winner in IRVElims:
        proofString = makeProofString(winner, Fore.RED)
        print(
            proofString
            + "IRV {0:2d}:".format(IRVElims.index(winner))
            + Fore.BLACK
            + " Candidate "
            + str(winner[0])
            + " cannot be eliminated next when "
            + (str(winner[1]) if winner[1] else "{}")
            + " are eliminated."
        )


def makeProofString(assertionTriple, colourString):
    if assertionTriple[2]:
        return "Confirmed:   "
    else:
        return colourString + "Unconfirmed: "


# Build printable pretty trees.
def buildPrintedResults(
    apparentWinner, apparentNonWinnersWithNames, WOLosers, IRVElims
):
    elimTrees = []
    apparentNonWinners = [c[0] for c in apparentNonWinnersWithNames]
    for c in apparentNonWinnersWithNames:
        candidateSet = set(apparentNonWinners).copy()
        candidateSet.add(apparentWinner)
        candidateSet.remove(c[0])
        treeAsLists = buildRemainingTreeAsLists(c[0], candidateSet, WOLosers, IRVElims)
        treeAsTuples = treeListToTuple(treeAsLists)
        drawnTree = svgling.draw_tree(treeAsTuples)
        elimTrees.append(
            Caption(drawnTree, "Pruned tree in which " + printTuple(c) + " wins.")
        )
    return elimTrees


# Takes the output of BuildPrintedResults and pretty-prints them one below the other along the page
def printTrees(elimTrees):
    if len(elimTrees) == 0:
        print("Error: printTrees received an empty list of trees.")
        # VT: think about whether to exit here.

    if len(elimTrees) == 1:
        return elimTrees[0]

    # if len(elimTrees)==2:
    #    return RowByRow(elimTrees[0],elimTrees[1])

    return RowByRow(elimTrees[0], printTrees(elimTrees[1:]))
