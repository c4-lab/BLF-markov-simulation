import numpy as np
import scipy.stats as stats
import random

from config import *

class Constants:


    def __init__(self):
        np.random.seed(seed=seed_val[0])
        random.seed(seed_val[0])

        self.coherence_matrix = self.__set_coh_matrix(number_of_bits)


    def __set_coh_matrix(self, k):

        m = np.zeros((2**k,k))

        for r_index, row in enumerate(m):
            row_decay_vals = []
            p_i_minus_1 = None
            for c_index, col in enumerate(row):
                if c_index == 0:
                    p_i = random.uniform(coh_mat_min_value, coh_mat_max_value)
                else:
                    p_i = p_i_minus_1 /2

                p_i_minus_1 = p_i
                row_decay_vals.append(p_i)

            random.shuffle(row_decay_vals)
            m[r_index] = row_decay_vals

        return m


    def get_coh_matrix(self):
        return self.coherence_matrix
    
