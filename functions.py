import arviz as az
import numpy as np 

def get_mode(all_chains, bw='silverman'): 
    '''Function for marginal mode estimation, based on arviz package'''
    mode_est = []
    for i in range(all_chains.shape[1]):
        kdes = az.kde(all_chains[:,i].flatten(), bw=bw)
        mest = kdes[0][np.where(kdes[1] == max(kdes[1]))[0][0]]
        mode_est.append(mest)
    mode_est = np.array(mode_est)
    return(mode_est)