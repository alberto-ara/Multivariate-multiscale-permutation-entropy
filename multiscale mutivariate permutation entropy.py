import itertools
import numpy as np
from pyentrp import entropy as ent
import matplotlib.pyplot as plt

def mpe(mts, m, d):
    """
    This function computes the multivariate permutation entropy of a multivariate time series. 
    The algorithm follows the formulations of Morabito et al. (2012) and it is based for the most part on
    Nikolay Donets' pyEntropy code (2012).
        
    INPUTS are:
        - mts: multivariate time series of shape "N series x N samples"
        - m: order of possible permutation motifs
        - d: time-lag
        
    OUTPUTS are:
        - pe_channel: single series entropy
        - pe_cross: cross-series entropy
        
    References:
        - Morabito, F.C., Labate, D., La Foresta, F., Bramanti, A., Morabito, G. & Palamara, I. (2012). 
          Multivariate Multi-Scale Permutation Entropy for Complexity Analysis of Alzheimer’s Disease EEG.
          Entropy, 14, 1186-1202.
        - Donets, N. (2013). PyEntropy. Github repository, https://github.com/nikdon/pyEntropy
    """
    # initialize parameters
    n = len(mts[0])
    e = len(mts)
    permutations = np.array(list(itertools.permutations(range(m))))
    t = n - d * (m - 1)
    c = []
    p = []
    pe_channel = []
    
    for j in range(e):
        c.append([0] * len(permutations))
        
    # compute single series permutation entropy based on the multivariate distribution of motifs
    for f in range(e):
        for i in range(t):
            sorted_index_array = np.array(np.argsort(mts[f][i:i + d * m:d], kind='quicksort'))
            for j in range(len(permutations)):
                if abs(permutations[j] - sorted_index_array).any() == 0:
                    c[f][j] += 1
    
        p.append(np.divide(np.array(c[f]), float(t*e)))
        pe_channel.append(-np.nansum(p[f] * np.log2(p[f])))
    
    # compute the cross-series permutation entropy based on the multivariate distribution of motifs
    rp = []
    pe_cross = []
    for w in range(len(permutations)):
        rp.append(np.nansum(np.array(p)[:,w]))
    
    pe_cross = -np.nansum(rp * np.log2(rp))
        
    return pe_channel, pe_cross

def mmpe (mts, m, d, s):
    """
    This function attempts to compute the multivariate multiscale permutation entropy of a multivariate 
    time series. The algorithm follows the formulations of Morabito et al. (2012) and it is based for the 
    most part on Nikolay Donets' pyEntropy code (2012).
    
    INPUTS are:
        - mts: multivariate time series of shape "N series x N samples"
        - m: order of possible permutation motifs
        - d: time-lag
        - s: scale factor
        
    OUTPUTS are:
        - ms: list of differently scaled multivariate permutation entropies
        
    References:
        - Morabito, F.C., Labate, D., La Foresta, F., Bramanti, A., Morabito, G. & Palamara, I. (2012). 
          Multivariate Multi-Scale Permutation Entropy for Complexity Analysis of Alzheimer’s Disease EEG.
          Entropy, 14, 1186-1202.
        - Donets, N. (2013). PyEntropy. Github repository, https://github.com/nikdon/pyEntropy
    """
    
    # get coarsed series and compute the multivariate permutation entropy on each scale
    ms = []
    for a in range(s):
        coarse_time_series = []
        for f in range(len(mts)):
            coarse_time_series.append(ent.util_granulate_time_series(mts[f], a + 1))
        
        pe = mpe(coarse_time_series, m, d)
        ms.append(pe)

    return ms
