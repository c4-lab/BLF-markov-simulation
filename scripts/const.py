import numpy as np
import scipy.stats as stats
import random

import config
import utilities

class Constants:




    def __init__(self):
        
        np.random.seed(seed=config.seed_val[0])
        random.seed(config.seed_val[0])

        print("Woking with bits to flip: {} - {}".format(config.coh_mat_min_bits_to_flip,config.coh_mat_max_bits_to_flip))
        #self.coherence_matrix = self.__set_coh_matrix(config.number_of_bits)
        self.coh_transition_matrix = self.__set_transition_matrix(config.number_of_bits,config.alpha_d)
        self.bit_matrix = self.__set_bit_matrix(config.number_of_bits)



    def __prune_matrix(self,m):
        m = m * (m > config.min_prob)
        row_sums = m.sum(axis = 1)
        return m / row_sums[:,np.newaxis]



    def __set_transition_matrix(self, k, a_d):
        """ Generates a transition matrix using a symmetric dirichlet"""
        result = np.random.dirichlet([a_d]*(2**k),2**k)

        if config.cliff:
            return self.__prune_matrix(result)
        else:
            return result

    def __set_bit_matrix(self,k):
        m = np.zeros((2**k,k))
        for r_index, row in enumerate(m):
            m[r_index] = utilities.int2bool(r_index,k)
        return m


    def __set_coh_matrix(self, k):

        m = np.zeros((2**k,k))

        for r_index, row in enumerate(m):
            row_decay_vals = []
            p_i_minus_1 = None
            bits_to_flip = random.randint(config.coh_mat_min_bits_to_flip,config.coh_mat_max_bits_to_flip)
            for c_index in range(bits_to_flip):
                if c_index == 0:
                    p_i = random.uniform(config.coh_mat_min_value, config.coh_mat_max_value)
                else:
                    p_i = p_i_minus_1 /2

                p_i_minus_1 = p_i
                row_decay_vals.append(p_i)

            row_decay_vals+= [0]*(len(row)-bits_to_flip)
            random.shuffle(row_decay_vals)
            m[r_index] = row_decay_vals

        return m


    def get_coh_matrix(self):
        return self.coherence_matrix

    def get_coh_tx_matrix(self):
        return self.coh_transition_matrix

    def get_bit_matrix(self):
        return self.bit_matrix



