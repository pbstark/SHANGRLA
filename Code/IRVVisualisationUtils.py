import svgling
from svgling.figure import Caption, SideBySide, RowByRow

# Convert a tree in list form into the same tree in tuple form suitable for
# svgling.
def treeListToTuple(t):
    # If t is an empty list, we shouldn't have got this far.
    if not t:
        warn("Error: empty list in tree drawing")
    # Leaf.  Return the name of the candidate and the assertions we've excluded it with.
    tag=""
    # TODO: look at whether t[0][1][1] is true (proved) or false and colour accordingly.
    if len(t) == 1:
        if t[0][1]:
            tag += "NEB "+str(t[0][1])
        if t[0][1] and t[0][2]:
            tag +="\n"
        if t[0][2]:
            tag += "NEN "+str(t[0][2])
        # print("Built tag"+str(t[0][0])+tag)
        return((t[0][0],tag)) 
    # Otherwise recurse.
    else:
        tList = []
        for branch in t[1]:
            tList.append(treeListToTuple(branch))
        return ((t[0],)+tuple(tList))

def parseAssertions(auditfile,candidatefile):
    #FIXME: Hardcoded to just look at the first audit for now.
    audit = auditfile["audits"][0]
    apparentWinner = audit["winner"]
    apparentWinnerName = findCandidateName(apparentWinner,candidatefile)
    print("Apparent winner: "+apparentWinner+", "+apparentWinnerName)
    apparentNonWinners=audit["eliminated"]
    apparentNonWinnersWithNames = findListCandidateNames(apparentNonWinners, candidatefile)
    print("Apparently eliminated: "+str(apparentNonWinnersWithNames))
    print("\n")
    assertions = audit["assertions"]

    # WOLosers is a list of tuples - the first element of the tuple is the loser,
    # the second element is the set of all the candidates it loses relative to.
    WOLosers = []

    # IRVElims is also a list of tuples - the first element is the candidate,
    # the second is the set of candidates already eliminated.
    # An IRVElim assertion states that the candidate can't be the next
    # eliminated when the already-eliminated candidates are exactly the set
    # in the second element of the tuple.
    IRVElims = []

    for a in assertions:
        
        if ("proved" in a) and (a["proved"]=="True"):
            proved = True
        else:
            proved = False
            
        if a["assertion_type"]=="WINNER_ONLY":
            if a["already_eliminated"] != "" :
                # VT: Not clear whether we should go on or quit at this point.
                warn("Error: Not-Eliminated-Before assertion with nonempty already_eliminated list.")
                
            l = a["loser"]
            w = a["winner"]


            WOLosers.append((l,w,proved))
                    
        if a["assertion_type"]=="IRV_ELIMINATION":
            l = a["winner"]
            IRVElims.append((l,set(a["already_eliminated"]),proved))
    return((apparentWinner,apparentWinnerName), apparentNonWinners, WOLosers, IRVElims)


# Given a candidate ID, find their name in the Candidate Manifest.
def findCandidateName(ID,candidateFile):
    candidates = candidateFile["List"]
    for c in candidates:
        if str(c["Id"])==ID:
            return c["Description"]
    
    return ""

# Given a list of candidate IDs, build a list of (ID, Name) tuples.
def findListCandidateNames(IDList,candidateFile):
    return list(map(lambda id : (id,findCandidateName(id,candidateFile)),IDList))

# This takes a root candidate c and a set S of candidates still to
# be built in to the tree (i.e. those to be eliminated earlier, closer to the leaves)
# it checks whether any assertions apply to this point in the elimination and, if so,
# prunes the tree here.
def buildRemainingTreeAsLists(c,S,WOLosers,IRVElims):

    # If c is in the list of candidates yet to be eliminated, this is a bug.
    if c in S:
        print("Error: c is in S.  c = "+str(c)+". S = "+str(S)+".\n")
    # if S is empty, return the leaf
    # Note that this indicates an error in the RAIRE audit
    # process - we're producing a tree 
    # in which we haven't pruned all the leaves.  
    # The intention is to make it visually obvious to an auditor.
    # Hence the ***
    if not S:
        return [[c, "***Unpruned leaf - ", 
            "RAIRE assertions do not exclude all other winners!***"]]

    pruneThisBranch = False
    NEBTags = []
    NENTags = []
    
    # if c is a loser defeated by a candidate in S, prune here.
    # Tag with NEB assertion number we used to prune.
    for loser in WOLosers:
        if c==loser[0] and ((loser[1] in S)):
            pruneThisBranch = True
            NEBTags.append((WOLosers.index(loser),loser[2]))
    
    # if c cannot be eliminated by IRV in exactly the case where S is the already-eliminated set, 
    # prune here.  Tag with NEN assertion number we used to prune.
    for winner in IRVElims:
        if c==winner[0] and winner[1]==S:
            pruneThisBranch = True
            NENTags.append((IRVElims.index(winner),winner[2]))

    if pruneThisBranch:
    # Base case: if we prune here, tag it with all the assertions
    # that could be used to prune.
        tree=[[c,NEBTags,NENTags]]        
    else:
    # if we didn't prune here, recurse        
        tree=[c,[]]
        for c2 in S:
            smallerSet = S.copy()
            smallerSet.remove(c2)
            tree[1].append(buildRemainingTreeAsLists(c2,smallerSet,WOLosers,IRVElims))
    
    return tree

def printAssertions(WOLosers,IRVElims):    
    if len(WOLosers) != 0:
        print("Not-Eliminated-Before assertions: ")
    for loser in WOLosers:
        print('NEB {0:2d}: Candidate '.format(WOLosers.index(loser))+str(loser[1])+' cannot be eliminated before '+str(loser[0])+'.')
    
    if len(IRVElims) != 0:
        print("\n")
        print("Not-Eliminated-Next assertions: ")
    for winner in IRVElims:
        print('NEN {0:2d}: Candidate '.format(IRVElims.index(winner))+str(winner[0])+' cannot be eliminated next when '+str(winner[1])+' are eliminated.')


# Build printable pretty trees.
def buildPrintedResults(apparentWinner, apparentNonWinners, WOLosers,IRVElims):
    elimTrees=[]
    for c in apparentNonWinners:
        candidateSet=set(apparentNonWinners).copy()
        candidateSet.add(apparentWinner)
        candidateSet.remove(c)
        treeAsLists=buildRemainingTreeAsLists(c,candidateSet, WOLosers, IRVElims)
        treeAsTuples=treeListToTuple(treeAsLists)
        drawnTree = svgling.draw_tree(treeAsTuples)
        elimTrees.append(Caption(drawnTree,"Pruned tree in which "+c+" wins."))
    return elimTrees

# Takes the output of BuildPrintedResults and pretty-prints them one below the other along the page
def printTrees(elimTrees):
    if len(elimTrees)==0:
        print("Error: printTrees received an empty list of trees.")
        #VT: think about whether to exit here.
        
    if len(elimTrees)==1:
        return elimTrees[0]
    
    #if len(elimTrees)==2:
    #    return RowByRow(elimTrees[0],elimTrees[1])
    
    return RowByRow(elimTrees[0],printTrees(elimTrees[1:]))
        
    
