# RAIRE Users Guide Test

This page describes step-by-step instructions for running an audit of an IRV election.


The theory is described in *[RAIRE: Risk-Limiting Audits for IRV Elections](https://arxiv.org/abs/1903.08804)* by Michelle Blom, Peter J. Stuckey, Vanessa Teague.  A simple example is *[here](code/RAIREExample.ipynb)*

These are the steps for running the audit:


1. Download the electronic Cast Vote Records (CVRs) from the *[San Francisco Department of Elections](https://sfelections.sfgov.org/results)*


2. Translate them into RAIRE's format using the CVR converter at **** 


3. Download the *[RAIRE tool](https://github.com/michelleblom/audit-irv-cp)* and compile it.


4. Run RAIRE on the reformatted CVRs from Step 2.  RAIRE will output a list of Assertions.


5. Upload RAIRE's assertions to the auditing notebook here, along with the electronic CVRs (from Step 1) and election metadata as requested.


6. Enter an appropriately-generated seed as requested.


7.  Download the list of ballot IDs for audit and upload them at *[the Paper Ballot checking tool](https://rla.vptech.io)*


8. Multiple teams may now retrieve and record the paper ballots they are instructed to find.  (**Perhaps insert instructions for the two-person protocol here.)  This produces a set of Manual Vote Records (MVRs).


9. Upload the MVRs here as instructed.  Execute the Risk-Limiting Audit calculation as instructed.


10.  If all the assertions are accepted the audit is now complete, and confirms the announced election outcome.  If not, decide whether to perform a full manual recount or follow the escalation instructions (returning to Step 7). 

