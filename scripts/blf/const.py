import numpy as np
import scipy.stats as stats
import scipy.spatial.distance as distance
import random

import utilities

class Constants:




    def __init__(self,config):
        
        np.random.seed(seed=config.seed_val)
        random.seed(config.seed_val)

        self.bit_matrix = self.__set_bit_matrix(config.number_of_bits)
        self.hamming_matrix = self.__set_hamming_matrix(config.number_of_bits)



    def __set_bit_matrix(self,k):
        m = np.zeros((2**k,k))
        for r_index, row in enumerate(m):
            m[r_index] = utilities.int2bool(r_index,k)
        return m

    def __set_hamming_matrix(self,k):
        inertia_matrix = np.zeros((2**k,2**k))
        for row_st, row in enumerate(inertia_matrix):
            for col_st, col in enumerate(row):
                bits_difference = utilities.hamming(row_st, col_st)
                inertia_matrix[row_st, col_st] = k - bits_difference
        return inertia_matrix


    def get_bit_matrix(self):
        return self.bit_matrix



