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
[RAIRE assertion-generator](https://github.com/michelleblom/audit-irv-cp).  This produces a set of assertions sufficient to prove that the announced winner truly won.  Observed paper ballots can be entered using [Dan King and Laurent Sandrolini's tool for the San Francisco Election board](https://rla.vptech.io/home).

We provide an open-source reference implementation and exemplar calculations in Jupyter notebooks.

----------------------------------------------------

## Step-by-step guide to San Francisco RLA pilots

----------------------------------------------------

You can recompute for yourself every step of the San Francsicso 2019 RLA pilot audit, except the retrieval of paper ballots.

1. Download the latest cvr export zip file from 
https://sfelections.sfgov.org/november-5-2019-election-results-detailed-reports

Unzip it.

2. Open  https://github.com/pbstark/SHANGRLA/blob/master/ConvertCVRToRAIRE.html
in your web browser, enter the files you just unzipped, set:
- Check 'Suppress any marks with IsAmbiguous=true'
- Check 'Suppress any marks with a write in with no other mark at same rank'
- Do not check ''Include all pieces of paper in the RAIRE output'

- Do not check 'Election Day'
- Check 'Vote by Mail'

- Check only the DA's race (Contest 339).

Scroll to the bottom of the screen and click on 'Download RAIRE format' (give it a little while to run – it's a big file).

Save the file as [mydataDirectory]/SF-VBM-DA.raire

The file used in the SF pilot audit is based on their Preliminary Report 12:  https://github.com/pbstark/SHANGRLA/blob/master/Code/Data/SFDA2019_PrelimReport12VBMJustDASheets.raire

Optional test: if you go back to ConvertToRAIRE.html and check the 'Election Day' box (also leaving VBM checked), your results for every contest should match exactly with the 'short report' results at sfelections.sfgov.org. 

3. Download the [RAIRE assertion-generator](https://github.com/michelleblom/audit-irv-cp), compile then run it on the SF-VBM-DA.raire file.  For a 5% risk limit, the command is:
./irvaudit -rep_ballots [mydataDirectory]/SF-VBM-DA.raire -r 0.05 -agap 0.0 -alglog -simlog -json [mydataDirectory]/SF-VBM-DA.json

This gives you the assertions to test.

The file used in the SF pilot audit is 
https://github.com/pbstark/SHANGRLA/blob/master/Code/Data/SFDA2019_PrelimReport12VBMJustDASheetsAssertions.json

4. Pull https://github.com/pbstark/SHANGRLA and

Open assertion-RLA.ipynb somewhere that notebooks run, for example in Binder.
The default input files are set up for San Francisco Preliminary Results Report 12, including the ballot manifest 14.

Execute the notebook through to [23], which generates sample.csv
This is the list of ballots to be fetched for audit.

The default is the official seed of 93686630803205229070.  This has been used to generate all the test samples, either for 187 ballots (the estimated sample size) or 200 ballots (hardcoded for the pilot).  If you use this seed you can use the test mvr files; if you generate your own seed you need to generate your own Manual Vote Records via Steps 5 and 6.

5.  Download sample.csv.  This is
the list of ballot IDs to be fetched for audit.

6. Run *[the manual vote recorder](https://github.com/dan-king/RLA-MVR)* on sample.csv to generate the *manual vote records* mvr.json.  You can also use *[the hosted version](https://rla.vptech.io)*
This is the part where paper ballots are fetched.  If you don't have access to the paper ballots you need to simulate this step.

7. Upload mvr.json to your notebook.  (This assumes you can upload files to wherever your notebook is running.  If you're using Binder, for example, this is not possible and you can only use the test mvr files that are in the github repo.)

8.  Keep running assertion-RLA.ipynb.  Read the computed p values in step [26].  If they are all less than 0.05 it should say AUDIT COMPLETE and we are done.  If they're not, this means a larger sample is required. For the pilot, no escalation was required.

9.  Run RAIREExampleDataParsing.ipynb and view the elimination trees.  Check that all the branches with an alternative winner are pruned.
