import numpy as np
import scipy.stats as stats
import scipy.spatial.distance as distance
import random

import config
import utilities

class Constants:




    def __init__(self):
        
        np.random.seed(seed=config.seed_val[0])
        random.seed(config.seed_val[0])

        self.bit_matrix = self.__set_bit_matrix(config.number_of_bits)




    def __set_bit_matrix(self,k):
        m = np.zeros((2**k,k))
        for r_index, row in enumerate(m):
            m[r_index] = utilities.int2bool(r_index,k)
        return m


    def get_coh_tx_matrix(self):
        return self.coh_transition_matrix

    def get_bit_matrix(self):
        return self.bit_matrix



