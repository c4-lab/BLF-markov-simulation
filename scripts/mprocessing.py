
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
from collections import Counter
from random import choice

import time



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
        self.neighbors = {} # neighbors will also be dictionary of agents
        self.name = name
        self.tau = tau # agent's threshold defined in initialization

        if initial_state: # if provided with initial knowledge state
            if isinstance(initial_state, list):
                self.initial_state = initial_state
                self.knowledge_state = initial_state
            else:
                raise ValueError('list expected')
        self.dissonance_lst = None
        self.next_state_probs = None
        self.next_state_onehot = None
        self.next_state = None
        self.soc_probs = None
        self.state_disagreements = None

    def reset_state(self,nstate = None):
        if nstate is not None:
            self.initial_state = nstate
        self.knowledge_state = self.initial_state

    def add_neighbors(self, neighbor_agent):
        """add neigbhor agent to the dictionary of neighbors
        which has neighbor agent name for key and agent itself as value"""
        self.neighbors[neighbor_agent.name] = neighbor_agent
        neighbor_agent.neighbors[self.name] = self # neighbors also get new neighbors

    def get_neighbors(self):
        """return all neighbors for the agent"""
        return self.neighbors

    def get_neighbors_name(self):
        return [n.name for k, n in self.neighbors.items()]

    def remove_neighbors(self, name):
        """remove neighbor for the agent with name"""
        if name in self.neighbors:
            del self.neighbors[name]
            print('removed neighbor')
        else:
            print('neighbor not found')


    def __str__(self):
        print('-'*30)
        s = 'agent name : ' + self.name
        s += '\nknowledge_state: [' + ', '.join(map(str, self.knowledge_state)) + ']'
        for n,v in self.neighbors.items():
            s += '\n'
            s += self.name + ' <-> ' + v.name

        return s


# In[4]:


def update_knowledge(agt):
        """Takes environment with common values to compute"""
        # first convert state binary to int to get the row in coherence matrix
        txn = worker_env.coherence_matrix
        bit_matrix = worker_env.bit_matrix
        alpha = worker_env.alpha

        # for agt in population:
        row_ptr = utilities.bool2int(agt.knowledge_state)
            # get the corresponding probabilites from the matrix
        coh_prob_tx = txn[row_ptr]
        ones_list = np.zeros(number_of_bits)
        dissonance_list = []
        disagreements = []

        for index, curr_bit_state in enumerate(agt.knowledge_state):
            # now look for neighbors who disagree in this bit value

            neigh_disagreement_count = count_dissimilar_neighbors(agt, index)

            # compute d as (# of neighbors disagree on bit/# of neighbors)
            if len(agt.neighbors) > 0:
                d = neigh_disagreement_count/len(agt.neighbors)
            else:
                d = 0

            #TODO: Handle the viral parameter - in general, if d = 0 and viral is set,
            #TODO: it should not be possible to make that transition

            if d > 0:
                dissonance = utilities.sigmoid(d, agt.tau)

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
        agt.next_state_probs = probs
        agt.soc_probs = soc_prob_tx
        agt.next_state = utilities.int2bool(np.random.choice(range(2**number_of_bits),1,p=probs)[0],number_of_bits)
#             print('Next State: ', agt.next_state)
        agt.dissonance_lst = dissonance_list
        agt.state_disagreements = disagreements
        return agt
#             return soc_prob_tx

def count_dissimilar_neighbors(agt, kbit):
    count = 0
    for name, agent in agt.neighbors.items():
        if agent.knowledge_state[kbit] != agt.knowledge_state[kbit]:
            count += 1

    return count



class Environment:
    """ Represents the context in which an agent operates
    This should provide any necessary bookkeeping for the agents, including
    functionality for managing id generation, and accessing other agents by id
    """

    def __init__(self, population, coherence, bit_mat, alpha):
        self.population = population
        """Holder for the population, indices correspond to ids"""

        self.coherence_matrix = coherence
        self.bit_matrix = bit_mat
        self.alpha = alpha

        """Perhaps some special variables that are shared by all agents"""

    def lookup(self, id: int) -> Agent:
        """ Get the agent corresponding to this id
        Args:
            id: The id of the agent to retrieve
        Returns:
            The Agent that corresponds to this index
        """
        return self.population[id]


def create_environment(proportion, parameter):

    # bits shuffling for equal bits
    count = 0
    states = [utilities.int2bool(random.randint(0, 63), number_of_bits) for i in range(num_agents)]
    tau_distr = get_tau_distr()
    list_agents = []

    selected_agents = random.sample(list(range(num_agents)), int(proportion*num_agents))
    for i in range(num_agents):

        in_state = [0,0,0,0,0,0]
        a = Agent(name='agent{}'.format(i), tau=random.choice(tau_distr), initial_state=in_state)
        list_agents.append(a)

    # create network
    G = nx.watts_strogatz_graph(num_agents, 10, parameter, seed=0) # FIX THIS! change rewire parameters as from different starting, 1 means random graph as each node is going to rewired and no structure is saved
    all_edges = G.edges()

    for edge in all_edges:
        list_agents[edge[0]].add_neighbors(list_agents[edge[1]])

    agt_stack = [list_agents[0]]
    count = 0
    num_needed = int(proportion*num_agents)
#     while len(agt_stack)>0:
    agt = choice(list_agents)
    agt_queue = [agt]
    visited = []
    while (count < num_needed) and agt_stack:
        agt = agt_stack.pop(0)
        if agt.name not in visited:
            agt.reset_state([1,1,1,1,1,1])
            visited.append(agt.name)
            count += 1
            for _, ngh in agt.get_neighbors().items():
                agt_stack.append(ngh)
#     nx.draw(G, with_labels=True)
    return list_agents, G


def get_network_df(list_agents):
    network_df = pd.DataFrame({'Agent Name':[], 'Neighbors':[]})
    for agt in list_agents:
        neighbors = agt.get_neighbors_name()
        network_df = network_df.append({'Agent Name':agt.name,
                                        'Neighbors':neighbors}, ignore_index=True)
    return network_df



def initializer(arg:Environment) -> None:
    """  Initializes the worker context
    The initializer function takes care of copying any data from the Master's context into
    the Worker's context. This copy of the data is is made available to each worker via a global
    variable
    Args:
        arg: In this case, this is just the Environment
    """
    global worker_env
    worker_env = arg


def run_simulation(alpha, coherence, bit_mat, end_time, proportion, parameter):

    list_agents, network_graph = create_environment(proportion, parameter)
    environment = Environment(list_agents, coherence, bit_mat, alpha)
    # get network of the agents
    agent_network_df = get_network_df(list_agents)

    d = []
    generations = 0

    for t in range(end_time):
        print(t)
        # compute next state for all agents
#         for agt in list_agents:
#             soc_mat = agt.update_knowledge(alpha, coherence, bit_mat)

#         start = time.time()
#         end = time.time()

        with Pool(3, initializer, initargs=(environment,)) as p:
            list_agents = p.map(update_knowledge, environment.population)

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

    return pd.DataFrame(d), list_agents, network_graph, environment



def hamming(a,b):
    return(bin(a^b).count("1"))


if __name__ == '__main__':


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

    attrctr, inert, coh = analysis.init_coherence_matrix(number_of_bits, attrctrs_1, 3)


    ### This cell is for generating the dataset
    # constants intialization

    network_parameters = [0.7] #np.arange(0, 1, 0.1).round(2)
    proportion_parameters = [0.7]#np.arange(0, 1, 0.1).round(2)
    end_simulation_time = 10

    alphas = np.arange(0, 1, 0.1).round(2)
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

                    tmp_record_df, soc_mat, list_agents, network_graph = run_simulation(alpha, coh, bit_mat, end_simulation_time, i, j)
    #                 tmp_record_df = tmp_record_df[tmp_record_df['Time']==49]
                    tmp_record_df['exp'] = exp
                    record_df_list.append(tmp_record_df)

    end = time.time()
    print((end-start)/60.0)


# In[ ]:


# x_df = pd.concat(record_df_list)#.to_csv('results.csv', index=False)
# x_df.to_csv('records_tmp.csv', index=False)


# In[ ]:


# x_df.head()


# In[ ]:


# x_df.groupby(['Agent_Number', 'Time', 'alpha', 'Proportion', 'Parameter', 'Current' , 'Next']).size().to_frame('Count').sort_values(by=['Current', 'Count'])\
#          .drop_duplicates(subset='Count', keep='last')


# In[ ]:


# run_simulation(alpha, coh, bit_mat, end_simulation_time, i, j)


# In[ ]:


# x_df[x_df['Agent_Number']==0][['Time','Current', 'Next']]


# In[ ]:
