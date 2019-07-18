import numpy as np
import scipy.stats as stats

shape = 8

class Constants:


    
    def __init__(self, k):
        self.coherence_matrix = np.random.rand(2**k,k)
        self.tau_distr = self.__set_tau_distr()
    
    def __set_tau_distr(self):
        # https://stackoverflow.com/a/28013759/5916727
    
        lower = 0
        upper = 1
        mu = 0.5
        sigma = 0.1
        N = 1000
        
        samples = stats.truncnorm.rvs(
          (lower-mu)/sigma,(upper-mu)/sigma,loc=mu,scale=sigma,size=N)
        
        return samples
    
    def get_tau_distr(self):
        return self.tau_distr


    def get_coh_matrix(self):
        return self.coherence_matrix