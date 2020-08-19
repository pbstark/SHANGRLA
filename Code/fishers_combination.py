import numpy as np
import scipy as sp
import scipy.stats

import matplotlib.pyplot as plt

def fisher_combined_pvalue(pvalues):
    """
    Find the p-value for Fisher's combined test statistic

    Parameters
    ----------
    pvalues : array_like
        Array of p-values to combine

    Returns
    -------
    float
        p-value for Fisher's combined test statistic
    """
    if np.any(np.array(pvalues)==0):
        return 0
    obs = -2*np.sum(np.log(pvalues))
    return 1-scipy.stats.chi2.cdf(obs, df=2*len(pvalues))

def create_modulus(N1, N2, n1, n2):
    """
    The modulus of continuity for the Fisher's combined p-value. This function
    returns the modulus of continuity, as a function of the distance between two
    beta values. Uses kaplan_markov risk function to calculate modulus.

    risk_funs : list
        risk functions used for strata
    N1 : int
        ballots cast in stratum 1
    N2 : int
        ballots cast in stratum 2
    n1 : int
        ballots sampled from stratum 1
    n2 : int 
        ballots sampled from stratum 2

    """
    if N1 == 0:
        T1 = lambda delta: 0
    else:
        T1 = lambda delta: 2*n1*np.log(1 + (N1+N2)/N1*delta)

    if N2 == 0:
        T2 = lambda delta: 0
    else:
        T2 = lambda delta: 2*n2*np.log(1 + (N1+N2)/N2*delta)

    return lambda delta: T1(delta) + T2(delta)

def maximize_fisher_combined_pvalue(N1, N2, pvalue_funs, beta_test_count=10, modulus=None, \
    alpha=0.05, feasible_beta_range=(0, 1/2)):
    """
    Grid search to find the maximum P-value.

    Find the smallest Fisher's combined statistic for P-values obtained by testing 
    two null hypotheses at level alpha using data X=(X1, X2).

    Parameters
    ----------
    N1 : int
        size of stratum 1
    N2 : int
        size of stratum 2
    pvalue_funs : array_like
        functions for computing p-values. The observed statistics/sample and known 
        parameters should be plugged in already. The function should take the beta 
        allocation AS INPUT and output a p-value.
    beta_test_count : int
        number of betas to search within the calculated range. Default is 10.
    modulus : function
        the modulus of continuity of the Fisher's combination function.
        This should be created using `create_modulus`.
        Optional (Default is None), but increases the precision of the grid search.
    alpha : float
        Risk limit. Default is 0.05.
    feasible_beta_range : array-like
        lower and upper limits to search over beta.
        Optional, but a smaller interval will speed up the search.
    """
    assert len(pvalue_funs)==2

    # find range of possible beta
    if feasible_beta_range is None:
        feasible_beta_range = calculate_beta_range(N1, N2)

    (beta_lower, beta_upper) = feasible_beta_range

    test_betas = np.array(np.linspace(beta_lower, beta_upper, beta_test_count))
    stepsize = (beta_upper - beta_lower)/(beta_test_count + 1)

    fisher_pvalues = np.empty_like(test_betas)
    for i in range(len(test_betas)):
        pvalue1 = np.min([1, pvalue_funs[0](test_betas[i])])
        pvalue2 = np.min([1, pvalue_funs[1](test_betas[i])])
        fisher_pvalues[i] = fisher_combined_pvalue([pvalue1, pvalue2])
    
    pvalue = np.max(fisher_pvalues)
    alloc_beta = test_betas[np.argmax(fisher_pvalues)]

    # If p-value is over the risk limit, then there's no need to refine the
    # maximization. We have a lower bound on the maximum.

    if pvalue > alpha or modulus is None:
        return {'max_pvalue' : pvalue,
                'min_chisq' : sp.stats.chi2.ppf(1 - pvalue, df=4),
                'allocation beta' : alloc_beta,
                'tol' : None,
                'stepsize' : stepsize,
                'refined' : False
                }

    fisher_fun_obs = scipy.stats.chi2.ppf(1-pvalue, df=4)
    fisher_fun_alpha = scipy.stats.chi2.ppf(1-alpha, df=4)
    dist = np.abs(fisher_fun_obs - fisher_fun_alpha)
    mod = modulus(stepsize)

    if mod <= dist:
        return {'max_pvalue' : pvalue,
                'min_chisq' : fisher_fun_obs,
                'allocation beta' : alloc_beta,
                'stepsize' : stepsize,
                'tol' : mod,
                'refined' : False
                }
    else:
        beta_lower = alloc_beta - 2*stepsize
        beta_upper = alloc_beta + 2*stepsize
        refined = maximize_fisher_combined_pvalue(N1, N2, pvalue_funs, \
            beta_test_count=beta_test_count*10, modulus=modulus, alpha=alpha, 
            feasible_beta_range=(beta_lower, beta_upper))
        refined['refined'] = True
        return refined
    
def calculate_beta_range(N1, N2, upper_bound=1):
    '''
    Find the largest and smallest possible values of beta adjusted by the proportion 
    of stratum sizes.

    Input: 
    ------ 
    N_1 : int
        ballots cast in stratum 1
    N_2 : int
        ballots cast in stratum 2
    upper_bound : double
        upper bound on assorter function (dependent on social choice function)


    Returns: 
    --------
    (lb, ub): real ordered pair. lb is a lower bound on beta; ub is a upper bound

    Derivation:
    -----------
    Let A_s be the assorter mean in stratum s. The null hypothesis for stratum s is:
        A_s*N_s/N <= beta_s
        A_s <= beta_s*N/N_s

    Strict bounds:
    A_s is bounded by the range of non-negative values assigned by the assorter which
    is [0, 1]. Thus, beta_s must also have a strict lower bound of 0. By definition
    of the assertion, beta <= 1/2 so beta_s must have a strict upper bound of 1/2.

    Loose bounds:
    A result of testing the assertion A_s <= beta_s*N/N_s is that beta_s*N/N_s can be
    greater than 1 which will make the p-value 1. If one stratum has a large P-value,
    the other will have a small P-value. The combined P-value in these cases will
    likely not produce the maximum P-value. We can solve for a range of beta values
    in terms of stratum sizes that have the possibility of producing P-values not
    close to 0, 1.

    Let beta = beta_1 and beta_2 = 1/2-beta_1 to maximize the p-value for stratum 2.
    Let the assorter's upper bound be A_ub and N=N1+N2. 

    If N1 > N2, N1 has stricter bounds on the beta range:
        Stratum 1
            beta_1*N/N1 < A_ub
            beta_1 < A_ub*N1/N
        Stratum 2
            beta_2*N/N2 < A_ub
            (1/2-beta_1)*N/N2 < A_ub
            1/2-A_ub*N2/N < beta_1
        Thus, beta_1 is bounded by (1/2-A_ub*N2/N, A_ub*N1/N). 
        Combining with the sharp bounds
            beta_1 >= max(1/2-A_ub*N2/N, 0) = 1/2-A_ub*N2/N
            beta_1 <= min(A_ub*N1/N, 1/2) = 1/2

    If N2 > N1, N2 has stricter bounds on the beta range:
        Using the derivation above and switching the cases for stratum 1 and
        stratum 2, beta_2 can be defined as
            [1] beta_2 >= max(1/2-A_ub*N1/N, 0) = 1/2-A_ub*N1/N
            [2] beta_2 <= min(A_ub*N2/N, 1/2) = 1/2
        
        Solving for beta_1 (beta):
            [1] beta_2 >= 1/2-A_ub*N1/N
                1/2-beta_2 <= A_ub*N1/N
                beta <= A_ub*N1/N
            [2] beta_2 <= 1/2
                1/2-beta_2 >= 0
                beta >= 0
    '''
    return (1/2-upper_bound*N2/(N1+N2), 1/2) if N1>N2 else (0, upper_bound*N1/(N1+N2))

def plot_fisher_pvalues(N, pvalue_funs, beta_test_count=10, alpha=None, plot_strata=False):
    """
    Plot the Fisher's combined p-value for varying error allocations
    using data X=(X1, X2)

    Parameters
    ----------
    N : array_like
        Array of stratum sizes
    pvalue_funs : array_like
        functions for computing p-values. The observed statistics/sample and known
        parameters should be plugged in already. The function should take the beta
        allocation AS INPUT and output a p-value.
    beta_test_count : int
        number of betas to plot within the calculated range. Default is 10.
    alpha : float
        Optional, desired upper percentage point
    plot_strata : Boolean
        If True, plot will include p-values for the CVR and no CVR strata

    Returns
    -------
    plot : figure
        scatter plot of beta values against combined p-values
    """
    assert len(N)==2
    assert len(pvalue_funs)==2

    # find range of possible beta
    (beta_lower, beta_upper) = calculate_beta_range(N[0], N[1])

    betas = np.array(np.linspace(beta_lower, beta_upper, beta_test_count))

    fisher_pvalues = []
    cvr_pvalues = []
    nocvr_pvalues = []
    for b in betas:
        cvr_pvalues.append(np.min([1, pvalue_funs[0](b)]))
        nocvr_pvalues.append(pvalue_funs[1](b))
        fisher_pvalues.append(fisher_combined_pvalue([cvr_pvalues[-1], nocvr_pvalues[-1]]))

    plt.scatter(betas, fisher_pvalues, marker='x', color='black', label='Fisher combination')
    if plot_strata:
        plt.scatter(betas, cvr_pvalues, color='r', label='CVR')
        plt.scatter(betas, nocvr_pvalues, color='b', label='no CVR')
        plt.legend()
        
    if alpha is not None: 
        plt.axhline(y=alpha, linestyle='--', color='gray')
    plt.xlabel('Allocation of Error')
    plt.ylabel('Fisher Combined P-value')
    plt.ylim(0, 1)
    plt.show()

