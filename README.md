# Sets of Half-Average Nulls Generate Risk-Limiting Audits (SHANGRLA)

by Michelle Blom, Andrew Conway, Philip B. Stark, Peter J. Stuckey and Vanessa Teague. 

Risk-limiting audits (RLAs) offer a statistical guarantee: if a full manual tally of the paper ballots would show that the reported election outcome is wrong, an RLA has a known minimum chance of leading to a full manual tally.
RLAs generally rely on random samples.

With SHANGRLA we introduce a very general method of auditing a variety of election types, by expressing an apparent election outcome as a series of assertions.  
Each assertion is of the form "the mean of a list of non-negative numbers is
greater than 1/2."
The lists of nonnegative numbers correspond to _assorters_, which assign
a number to the selections made on each ballot (and to the cast vote record, for comparison audits).
Each assertion is tested using a sequential test of the null hypothesis that its complement holds.
If all the null hypotheses are rejected, the election outcome is confirmed. 
If not, we proceed to a full manual recount.
SHANGRLA incorporates several different statistical 
risk-measurement algorithms and extends naturally to plurality and super-majority 
contests with various election types including Range and Approval voting and Borda count.  

It can even incorporate Instant Runoff Voting (IRV) using the 
[RAIRE assertion-generator](https://github.com/michelleblom/audit-irv-cp).  Observed paper ballots can be entered using [Dan King and Laurent Sandrolini's tool for the San Francisco Election board](https://rla.vptech.io/home).

We provide an open-source reference implementation and exemplar calculations in Jupyter notebooks.



