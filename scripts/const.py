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
        self.reset_transition_matrix(config.tx_init_method)




    def __prune_matrix(self,m):
        if config.cliff:
            m = m * (m > config.min_prob)
        row_sums = m.sum(axis = 1)
        return m / row_sums[:,np.newaxis]



    def reset_transition_matrix(self, method):
        """ Generates a transition matrix using a symmetric dirichlet"""
        print("Initializing matrix using {} method again".format(method))
        m = self.__getattribute__("tx_{}_method".format(method))
        self.coh_transition_matrix = self.__prune_matrix(m())

    def tx_dirichlet_method(self):
        k = config.number_of_bits
        a_d = config.alpha_d
        return(np.random.dirichlet([a_d]*(2**k),2**k))

    def tx_bitwise_method(self):
        k = config.number_of_bits
        m = np.zeros((2**k,2**k))

        for r_index, row in enumerate(m):
            row_decay_vals = []

            bits_to_flip = random.randint(config.min_bits_to_flip,config.max_bits_to_flip)
            for c_index in range(bits_to_flip):
                if c_index == 0:
                    p_i = random.uniform(config.flip_min_value, config.flip_max_value)
                else:
                    p_i = p_i/2

                row_decay_vals.append(p_i)

            row_decay_vals+= [0]*(k-bits_to_flip)
            random.shuffle(row_decay_vals)
            probs = np.array(row_decay_vals)

            #Now figure out tx probabilities

            dest = np.array(range(2**k))
            src = (np.ones(2**k) * r_index).astype(np.int)

            #Here, flips just places a 1 in each position where the dest state differs from the src state
            flips = np.bitwise_xor(src,dest)

            #The following just sets up bit masks so we can extract the relevant probabilities here
            flips = np.array([list(np.binary_repr(x,k)) for x in flips]).astype(np.int)
            state_probs = np.prod(probs * flips + (1-probs)*(1-flips),1)
            m[r_index] = state_probs

        return(m)

    def hamming(self,a,b):
        return(bin(a^b).count("1"))


    def scale_by_stickbreaking(self,a):
        result = np.zeros(len(a))
        if type(a)==list:
            a = np.array(a)
        stick = 1.0
        for ix in (-1*a).argsort():
            p = a[ix]*stick
            result[ix] = p
            stick-=p
        return(result)


    def tx_bitblock_method(self):
        k = config.number_of_bits
        m = np.zeros((2**k,2**k))
        max_dist = k * (1+max(map(lambda x: x[1],config.attractors)))
        m_p = config.fa_max_prob
        m_f = config.fa_max_dist
        for r_index,row in enumerate(m):
            nrow = []
            for dest in range(2**k):
                dfrom = self.hamming(r_index,dest)
                dto = min(map(lambda x: (x[1]+1)*self.hamming(x[0],dest),config.attractors))
                p = max(0,(m_p - (dfrom * m_p/m_f))) * (1 - dto/max_dist)
                nrow.append(p)
            m[r_index] = self.scale_by_stickbreaking(nrow)
        #print(m)
        return(m)


    def tx_euclidean_method(self):
        k = config.number_of_bits
        m = np.zeros((2**k,2**k))
        max_dist = (k**.5) * (1+max(map(lambda x: x[1],config.attractors)))
        m_p = config.fa_max_prob
        m_f = config.fa_max_dist
        for r_index,row in enumerate(m):
            nrow = []
            for dest in range(2**k):
                dfrom = self.hamming(r_index,dest)
                dto = min(map(lambda x: (x[1]+1)*distance.euclidean(x[0],dest),config.attractors))
                p = max(0,(m_p - (dfrom * m_p/m_f))) * (1 - dto/max_dist)
                nrow.append(p)
            m[r_index] = self.scale_by_stickbreaking(nrow)
        #print(m)
        return(m)





    def __set_bit_matrix(self,k):
        m = np.zeros((2**k,k))
        for r_index, row in enumerate(m):
            m[r_index] = utilities.int2bool(r_index,k)
        return m


    def get_coh_tx_matrix(self):
        return self.coh_transition_matrix

    def get_bit_matrix(self):
        return self.bit_matrix



