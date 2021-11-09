from __future__ import annotations
import numpy as np
import pandas as pd
# from AgentClass import Agent
import const
import random
import networkx as nx
from config import number_of_bits, num_agents, tau_lower_bound, tau_upper_bound, tau_mu, tau_sigma, tau_n_samples
from scipy import stats
from collections import defaultdict
import json
import utilities
import copy

import matplotlib.pyplot as plt
import matplotlib
import seaborn as sns
from random import shuffle

import analysis

from scipy.spatial import distance

from multiprocessing import Pool
from multiprocessing.shared_memory import *
from collections import Counter
from random import choice
import sys


import time


glob_environment = None
glob_coherence_matrix = None
glob_population = None
glob_bit_matrix = None
glob_alpha = None
glob_buffers = []

def get_tau_distr():

    lower = tau_lower_bound
    upper = tau_upper_bound
    mu = tau_mu
    sigma = tau_sigma
    N = tau_n_samples

    samples = stats.truncnorm.rvs(
          (lower-mu)/sigma,(upper-mu)/sigma,loc=mu,scale=sigma,size=N)

    return samples




"""Each agent has coherence matrix and neighbors which are agents as well"""
import utilities
import numpy as np
import networkx
from scipy.stats import logistic
from config import contagion_mode, scale,number_of_bits
import utilities
from const import Constants

class Agent:

    def __init__(self, name='', tau=-1, knowledge_dim = 3, initial_state=None):
        self.neighbors = set() # neighbors are a list a of indices of other agents
        self.name = name
        self.tau = tau # agent's threshold defined in initialization

        if initial_state: # if provided with initial knowledge state
            if isinstance(initial_state, list):
                self.initial_state = initial_state
                self.knowledge_state = initial_state
            else:
                raise ValueError('list expected')
        # self.dissonance_lst = None
        # self.next_state_probs = None
        # self.next_state_onehot = None
        # self.next_state = None
        # self.soc_probs = None
        # self.state_disagreements = None


    def reset_state(self,nstate = None):
        if nstate is not None:
            self.initial_state = nstate
        self.knowledge_state = self.initial_state

    def add_neighbor_indices(self, *neighbors):
        """add neigbhor agent to the list of neighbor indices"""
        self.neighbors.update(neighbors)


    def get_neighbor_indices(self):
        return self.neighbors

    def update(self,environment):
        """Takes environment with common values to compute"""

        # first convert state binary to int to get the row in coherence matrix
        txn = environment.get_coherence_matrix()
        bit_matrix = environment.get_bit_matrix()
        alpha = environment.get_alpha()

        # for agt in population:
        row_ptr = utilities.bool2int(self.knowledge_state)

        # get the corresponding probabilites from the matrix
        coh_prob_tx = txn[row_ptr]
        ones_list = np.zeros(number_of_bits)
        dissonance_list = []
        disagreements = []

        for index, curr_bit_state in enumerate(self.knowledge_state):
            # now look for neighbors who disagree in this bit value


            # compute d as (# of neighbors disagree on bit/# of neighbors)
            if len(self.get_neighbor_indices()) > 0:
                d = count_dissimilar_neighbors(self, index, environment)/len(self.get_neighbor_indices())
            else:
                d = 0

            #TODO: Handle the viral parameter - in general, if d = 0 and viral is set,
            #TODO: it should not be possible to make that transition

            if d > 0:
                dissonance = utilities.sigmoid(d, self.tau)

            else:
                dissonance = 0

            dissonance_list.append(dissonance)

            # keeping track of disagreement of bits/total neighbors
            disagreements.append(d)
            # transition probabilities given social pressure for moving to a state
            # with a '1' at this bit
            ones_list[index] = (1-dissonance if curr_bit_state else dissonance)

        zeros_list = 1-ones_list
        tmp_soc_mat = ones_list * bit_matrix  + zeros_list * (1-bit_matrix)

        # Probabilities for each state given social pressure
        soc_prob_tx = np.prod(tmp_soc_mat,1)
        #TODO logs soc_prob_tx for each agent at each time step

        probs = alpha * soc_prob_tx + (1-alpha)*coh_prob_tx
        next_state_probs = probs
        soc_probs = soc_prob_tx
        next_state = utilities.int2bool(np.random.choice(range(2**number_of_bits),1,p=probs)[0],number_of_bits)
        #             print('Next State: ', agt.next_state)
        dissonance_lst = dissonance_list
        state_disagreements = disagreements
        return next_state

    def advance(self):
        self.knowledge_state = self.next_state
        self.next_state = None

    def __str__(self):
        print('-'*30)
        s = 'agent name : ' + self.name
        s += '\nknowledge_state: [' + ', '.join(map(str, self.knowledge_state)) + ']'

        return s


# In[4]:




def count_dissimilar_neighbors(ego, kbit, env:Environment):
    count = 0
    for agent in ego.get_neighbor_indices():
        alter_state = env.lookup_agent_state(agent)
        if ego.knowledge_state[kbit] != alter_state[kbit]:
            count += 1

    return count



class Environment:
    """ Represents the context in which an agent operates
    This should provide any necessary bookkeeping for the agents, including
    functionality for managing id generation, and accessing other agents by id
    """

    def __init__(self):
        self.names = {}

    def setup_shared(self, population, coherence, bit_mat, alpha):
        global glob_alpha, glob_bit_matrix, glob_coherence_matrix, glob_population
        pop = np.array([a.knowledge_state for a in population])
        glob_coherence_matrix = self.make_shared(coherence,"coherence")
        glob_population = self.make_shared(pop,"population")
        glob_bit_matrix = self.make_shared(bit_mat,"bit_matrix")
        glob_alpha = self.make_shared(np.array([alpha]),"alpha")



    def make_shared(self,orig,name = "unknown"):
        global glob_buffers
        buf = SharedMemory(create=True, size=orig.nbytes)
        #Note - this line is critical, or else the buffer dissappears when we exit the function's scope!!!
        glob_buffers.append(buf)
        self.names[name] = [buf.name,orig.shape,orig.dtype]
        shared = np.ndarray(shape=orig.shape,dtype=orig.dtype,buffer=buf.buf)
        shared[:] = orig[:]
        return shared

    def reattach(self):
        global glob_alpha, glob_bit_matrix, glob_coherence_matrix, glob_population
        print("Reattach {} ".format(self.names))
        glob_coherence_matrix = self.attach_shared("coherence")
        glob_population = self.attach_shared("population")
        glob_bit_matrix = self.attach_shared("bit_matrix")
        glob_alpha = self.attach_shared("alpha")

    def attach_shared(self,name):
        global glob_buffers
        spec = self.names[name]
        shm = SharedMemory(name = spec[0])
        glob_buffers.append(shm)
        return np.ndarray(shape = spec[1],dtype = spec[2], buffer = shm.buf)

    def cleanup(self):
        global glob_alpha, glob_bit_matrix, glob_coherence_matrix, glob_population
        del glob_population, glob_coherence_matrix, glob_bit_matrix, glob_alpha
        for b in glob_buffers:
            b.close()
            b.unlink()

    def get_alpha(self):
        return glob_alpha[0]

    def get_coherence_matrix(self):
        return glob_coherence_matrix

    def get_bit_matrix(self):
        return glob_bit_matrix

    def lookup_agent_state(self,idx):
        return glob_population[idx]

    def update_agent_state(self,idx,array):
        glob_population[idx] = array

    def update_all_agents(self,agents):
        for i in range(len(agents)):
            self.update_agent_state(i,agents[i].knowledge_state)



def init_agents(parameter):

    # bits shuffling for equal bits
    count = 0
    states = [utilities.int2bool(random.randint(0, 63), number_of_bits) for i in range(num_agents)]
    tau_distr = get_tau_distr()
    list_agents = []

    for i in range(num_agents):

        in_state = [0,0,0,0,0,0]
        a = Agent(name='agent{}'.format(i), tau=random.choice(tau_distr), initial_state=in_state)
        list_agents.append(a)

    # create network
    G = nx.watts_strogatz_graph(num_agents, 10, parameter, seed=0) # FIX THIS! change rewire parameters as from different starting, 1 means random graph as each node is going to rewired and no structure is saved
#     nx.draw(G, with_labels=True)
    return list_agents, G


def setup_environment(list_agents, network:networkx.Graph, proportion, coherence, bit_mat, alpha):

    all_edges = network.edges()

    for edge in all_edges:
        list_agents[edge[0]].add_neighbor_indices(edge[1])

    agt_stack = [0]
    count = 0
    num_needed = int(proportion*num_agents)
    visited = []
    while (count < num_needed) and agt_stack:
        agt_idx = agt_stack.pop(0)
        if agt_idx not in visited:
            list_agents[agt_idx].reset_state([1,1,1,1,1,1])
            visited.append(agt_idx)
            count += 1
            for ngh in list_agents[agt_idx].get_neighbor_indices():
                agt_stack.append(ngh)

    env = Environment()
    env.setup_shared(list_agents, coherence, bit_mat, alpha)
    return env


def initializer(evt:Environment):
    global glob_environment
    glob_environment = evt
    print("Reattach shared memory")
    glob_environment.reattach()

def update_agent(agt:Agent):
    return agt.update(glob_environment)

def run_simulation(alpha, coherence, bit_mat, end_time, proportion, parameter):
    list_agents, network_graph = init_agents(parameter)
    env = setup_environment(list_agents,network_graph,proportion,coherence,bit_mat,alpha)
    # get network of the agents
    #agent_network_df = get_network_df(list_agents)

    d = []
    generations = 0

    for t in range(end_time):
        print(t)
        # compute next state for all agents
#         for agt in list_agents:
#             soc_mat = agt.update_knowledge(alpha, coherence, bit_mat)

#         start = time.time()
#         end = time.time()

        with Pool(15,initializer=initializer, initargs=(env,)) as p:
            results = p.map(update_agent, list_agents)

        for idx,agent in enumerate(list_agents):
            agent.reset_state(results[idx])
            env.update_agent_state(idx,agent.knowledge_state)

#         update_knowledge(environment)

        # keep record of current record and all other values
#         for agt in list_agents:
# #             print('Viewing: ', agt.next_state)
#             row = {'Agent_Number': int(agt.name.split('t')[1]),
#                    'Time':t,
#                    # at any time step we will need normalized how many neighbors disagree on bits
#                    'bits_disagreement':np.array(agt.state_disagreements),
#                    'Current_Knowledge_State':agt.knowledge_state,
#                    'Current': utilities.bool2int(agt.knowledge_state),
#                    'alpha':alpha,
#                    'Next': utilities.bool2int(agt.next_state),
#                    'Next_Knowledge_State':agt.next_state,
#                    'Proportion': proportion,
#                     'Parameter': parameter}

            # d.append(row)

        # now update all agents next state with computed next state
        # for agt in list_agents:
        #     agt.knowledge_state = agt.next_state
        #     agt.next_state = None
        #     agt.dissonance_lst = None

        generations+=1
    env.cleanup()
    return pd.DataFrame(d), list_agents, network_graph, env



def hamming(a,b):
    return(bin(a^b).count("1"))


def doit():
    attrctr1 = utilities.bool2int([1,1,1,1,1,1])
    attrctr2 = utilities.bool2int([0,0,0,0,0,0])

    attrctrs = [attrctr1, attrctr2]
    attractors = {}
    number_attractors = 0
    while  number_attractors< 2:
        attractor_state = attrctrs.pop()
        attractor_depth = random.randint(1, 4) # depth for each attractors is picked randomly
        attractor_radius = random.randint(1, 2)

        attractors[attractor_state] = {'depth': attractor_depth, 'radius': attractor_radius}
        number_attractors += 1

    attrctrs_1 = [[k, 100, 1] for k,v in attractors.items()]

    attrctr, coh = analysis.init_coherence_matrix(number_of_bits, attrctrs_1, 3)


    ### This cell is for generating the dataset
    # constants intialization

    network_parameters = [0.7] #np.arange(0, 1, 0.1).round(2)
    proportion_parameters = [0.7]#np.arange(0, 1, 0.1).round(2)
    end_simulation_time = 10

    #alphas = np.arange(0, 1, 0.1).round(2)
    alphas = [0.1]
    exp_times = 1

    constants = const.Constants()

    bit_mat = constants.get_bit_matrix()

    record_df_list = []

    start = time.time()

    for exp in range(1):
        for alpha in alphas:
            for j in proportion_parameters:
                # first create environment


                for i in network_parameters:

                    run_simulation(alpha, coh, bit_mat, end_simulation_time, i, j)
                    #                 tmp_record_df = tmp_record_df[tmp_record_df['Time']==49]
                    # tmp_record_df['exp'] = exp
                    # record_df_list.append(tmp_record_df)

    end = time.time()
    print((end-start)/60.0)

# if __name__ == '__main__':
#     doit()





