# Sets of Half-Average Nulls Generate Risk-Limiting Audits (SHANGRLA)

Risk-limiting audits (RLAs) offer a statistical guarantee: if a full manual
tally of the paper ballots would show that the reported election outcome is
wrong, an RLA has a known minimum chance of leading to a full manual tally.
RLAs generally rely on random samples.

With SHANGRLA we introduce a very general method of auditing a variety of
election types, by expressing an apparent election outcome as a series of
assertions.  
Each assertion is of the form "the mean of a list of non-negative numbers is
greater than 1/2."

The lists of nonnegative numbers correspond to _assorters_, which assign a
number to the selections made on each ballot (and to the cast vote record, for
comparison audits).  Each assertion is tested using a sequential test of the
null hypothesis that its complement holds.
If all the null hypotheses are rejected, the election outcome is confirmed.
If not, we proceed to a full manual recount.
SHANGRLA incorporates several different statistical risk-measurement algorithms
and extends naturally to plurality and super-majority contests with various
election types including Range and Approval voting and Borda count.

SHANGRLA implements the (https://projecteuclid.org/journals/annals-of-applied-statistics/volume-17/issue-1/ALPHA-Audit-that-learns-from-previously-hand-audited-ballots/10.1214/22-AOAS1646.short)[ALPHA supermartingale tests] and (https://arxiv.org/abs/2303.03335)[ONEAudit]. The current version can work with CVRs, with ONEAudit CVRs, or for ballot-polling audits.
It does not yet support stratified audits, but there are stubs for them.

It can even incorporate Instant Runoff Voting (IRV) using the
[RAIRE assertion-generator](https://github.com/michelleblom/audit-irv-cp).
This produces a set of assertions sufficient to prove that the announced winner
truly won.  Observed paper ballots can be entered using [Dan King and Laurent
Sandrolini's tool for the San Francisco Election
board](https://rla.vptech.io/home).

We provide an open-source reference implementation and exemplar calculations in
Jupyter notebooks.


## Installation

### Installing from GitHub

Main version:

```
pip install git+https://github.com/pbstark/SHANGRLA.git@main
```

Development version:

```
pip install git+https://github.com/dvukcevic/SHANGRLA.git@dev
```

### Installing from a local copy (in development mode)

Install just the code:

```
pip install -e .
```

Also include the optional dependencies for tests and examples:

```
pip install -e .[test,examples]
```

## Authors and contributors

The initial code was written by Michelle Blom, Andrew Conway, Philip B. Stark,
Peter J. Stuckey and Vanessa Teague.

Additional development by Amanda Glazer, Jake Spertus, Ian Waudby-Smith,
David Wu, Alexander Ek, Floyd Everest and Damjan Vukcevic.


## Licences

Copyright (C) 2019-2024  Philip B. Stark, Vanessa Teague, Michelle Blom,
Peter Stuckey, Ian Waudby-Smith, Jacob Spertus, Amanda Glazer,
Damjan Vukcevic, David Wu, Alexander Ek, Floyd Everest.


### Software

[![GNU AGPL][agpl-img]][agpl]  
The software, and documentation of the software, in this repository is provided
under the [GNU Affero General Public License][agpl] (AGPL).
You can redistribute and/or modify the software and documentation under the
terms of the AGPL as published by the Free Software Foundation, either version
3 of the License, or (at your option) any later version.

The software is distributed in the hope that it will be useful, but WITHOUT ANY
WARRANTY; without even the implied warranty of MERCHANTABILITY or FITNESS FOR A
PARTICULAR PURPOSE. See the AGPL for more details.
A copy of the AGPL is provided in `LICENSE`.

### Other files

[![Creative Commons License][cc-img]][cc]  
The other documents in this repository (not including the software and
documentation of the software) are provided under a [Creative Commons
Attribution-NoDerivs 4.0 International License][cc] (CC BY-ND 4.0).


[agpl]: https://www.gnu.org/licenses/agpl-3.0.en.html
[agpl-img]: https://www.gnu.org/graphics/agplv3-88x31.png

[cc]: https://creativecommons.org/licenses/by-nd/4.0/
[cc-img]: https://i.creativecommons.org/l/by-nd/4.0/88x31.png
