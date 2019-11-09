[![Binder](https://mybinder.org/badge.svg)](https://mybinder.org/v2/gh/pbstark/CORLA18/master?filepath=code%2Fsuite_toolkit.ipynb)

# Risk-Limiting Audits by Stratified Union-Intersection Tests of Elections (SUITE)

by Kellie Ottoboni, Philip B. Stark, Mark Lindeman, and Neal McBurnett

Risk-limiting audits (RLAs) offer a statistical guarantee: if a full manual tally of the paper ballots would show that the reported election outcome is wrong, an RLA has a known minimum chance of leading to a full manual tally.
RLAs generally rely on random samples.
Stratified sampling---partitioning the population of ballots into disjoint
strata and sampling independently from the strata---may simplify logistics or increase efficiency compared
to simpler sampling designs, but makes risk calculations harder.
We present SUITE, a novel RLA method for stratified samples.
SUITE tests all possible partitions of outcome-changing error across strata.
For each partition, SUITE combines P-values for each stratum's error into a combined
P-value.
(SUITE is agnostic about the methods for finding stratum-level P-values.) 
The combined P-value is maximized over all such partitions. 
The audit can stop if the maximum combined P-value is less than the risk limit.
SUITE is immediately useful in Colorado.
Voting systems in some Colorado counties (comprising 98.2% of voters)
allow auditors to check how the system interpreted each ballot, 
which allows efficient _ballot-level comparison_ RLAs.
(Other counties use _ballot polling_, which is less efficient.)
Extant approaches to conducting an RLA of a statewide contest would require Colorado to make major procedural changes, or would sacrifice the efficiency of ballot-level comparison.
In contrast, SUITE requires little change to Colorado's procedures and is substantially more efficient than
a statewide ballot-polling RLA.
The two strata comprise ballots cast in counties that can conduct ballot-level comparisons, and the rest.
Stratum-level P-values can be found using modifications of ballot-polling and ballot-level 
comparison, derived here.
We provide an open-source reference implementation and exemplar calculations in Jupyter notebooks.



[Fisher's combination method illustration](https://mybinder.org/v2/gh/pbstark/CORLA18/master?filepath=code%2Ffisher_combined_pvalue.ipynb)

[Example Notebook 1](https://mybinder.org/v2/gh/pbstark/CORLA18/master?filepath=code%2Fhybrid-audit-example-1.ipynb)

[Example Notebook 2](https://mybinder.org/v2/gh/pbstark/CORLA18/master?filepath=code%2Fhybrid-audit-example-2.ipynb)
