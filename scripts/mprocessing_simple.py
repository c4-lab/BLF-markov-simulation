# This is a purely abstraction free version of the ABM
# commenting line below still works fine
#from __future__ import annotations
import numpy as np
import ray
import const
import random
import networkx as nx
# all parameters imported from the config file
from config import number_of_bits, num_agents, tau_lower_bound, tau_upper_bound, tau_mu, tau_sigma, tau_n_samples, watts_strogatz_graph_param,sim_network_params_lst, end_sim_time, alpha_range, num_experiments
from scipy import stats
import analysis
import time
import utilities

shrd_static = {
    "coherence_matrix":None,
    "bit_matrix":None,
    "alpha":None
}

# def hamming(a,b):
#     return(bin(a^b).count("1"))



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

class Agent:

    def __init__(self, idx, tau=-1):
        self.idx = idx
        self.neighbors = set() # neighbors are a list a of indices of other agents
        self.tau = tau # agent's threshold defined in initialization

    def add_neighbor_indices(self, neighbors):
        """add neigbhor agent to the list of neighbor indices"""

        self.neighbors.update(neighbors)

    def update(self,static,dynamic):
        """Takes environment with common values to compute"""

        # first convert state binary to int to get the row in coherence matrix
        txn = static["coherence_matrix"]
        bit_matrix = static["bit_matrix"]
        alpha = static["alpha"]
        kstate = dynamic[self.idx]

        # for agt in population:
        row_ptr = utilities.bool2int(kstate)

        # get the corresponding probabilites from the matrix
        coh_prob_tx = txn[row_ptr]
        ones_list = np.zeros(number_of_bits)


        for kbit, curr_bit_state in enumerate(kstate):
            # now look for neighbors who disagree in this bit value
            count = 0
            for alter in self.neighbors:
                alter_state = dynamic[alter]
                if curr_bit_state != alter_state[kbit]:
                    count += 1

            dissonance = 0 if len(self.neighbors) == 0 else count/len(self.neighbors)

            #TODO: Handle the viral parameter - in general, if d = 0 and viral is set,
            #TODO: it should not be possible to make that transition

            if dissonance > 0:
                dissonance = utilities.sigmoid(dissonance, self.tau)

            # transition probabilities given social pressure for moving to a state
            # with a '1' at this bit
            ones_list[kbit] = (1-dissonance if curr_bit_state else dissonance)

        zeros_list = 1-ones_list
        tmp_soc_mat = ones_list * bit_matrix  + zeros_list * (1-bit_matrix)

        # Probabilities for each state given social pressure
        soc_prob_tx = np.prod(tmp_soc_mat,1)
        #TODO logs soc_prob_tx for each agent at each time step

        probs = alpha * soc_prob_tx + (1-alpha)*coh_prob_tx
        return utilities.int2bool(np.random.choice(range(2**number_of_bits),1,p=probs)[0],number_of_bits)



def init_agents(network:nx.Graph):


    tau_distr = get_tau_distr()
    list_agents = []

    for i in range(network.number_of_nodes()):
        a = Agent(idx = i, tau=random.choice(tau_distr))
        list_agents.append(a)
        n = [x for x in network.neighbors(i)]
        a.add_neighbor_indices(n)

    # create network
    return list_agents


def setup_environment(network:nx.Graph, coherence, bit_mat, alpha):
    list_agents = init_agents(network)
    shrd_static["coherence_matrix"] = coherence
    shrd_static["bit_matrix"] = bit_mat
    shrd_static["alpha"] = alpha

    values = list(range(coherence.shape[0]))
    random.shuffle(values)
    states = []
    for i in range(num_agents):
        states.append(utilities.int2bool(values[i % len(values)],number_of_bits))
    print(states[0])
    return list_agents, states


@ray.remote
def agent_update(agent:Agent, static, dynamic):
    return agent.update(static,dynamic)


def run_simulation(end_time, agents, states):
    static_obj = ray.put(shrd_static)
    for t in range(end_time):
        dynamic_obj = ray.put(states)
        results = [agent_update.remote(agent,static_obj,dynamic_obj) for agent in agents]
        states = ray.get(results)
        del dynamic_obj

def create_attractors(attractors_states=[]):
    if len(attractors_states)==0:
        attrctr1 = utilities.bool2int(1 for _ in range(number_of_bits))
        attrctr2 = utilities.bool2int(0 for _ in range(number_of_bits))

        attrctrs = [attrctr1, attrctr2]

    else:
        attrctrs = attractors_states[:]

    attractors = {}
    number_attractors = 0
    while  number_attractors< len(attrctrs):
        attractor_state = attrctrs.pop()
        attractor_depth = random.randint(1, 4) # depth for each attractors is picked randomly
        attractor_radius = random.randint(1, 2)

        attractors[attractor_state] = {'depth': attractor_depth, 'radius': attractor_radius}
        number_attractors += 1

    attrctrs_1 = [[k, 100, 1] for k,v in attractors.items()]
    attrctr, coherence_mat = analysis.init_coherence_matrix(number_of_bits, attrctrs_1, 3)

    return coherence_mat

def doit():
     # setting two attractors with one having all zeros and other with all ones


    network_parameters = sim_network_params_lst
    end_simulation_time = end_sim_time

    #alphas = np.arange(0, 1, 0.1).round(2)
    alphas = alpha_range
    constants = const.Constants()

    bit_mat = constants.get_bit_matrix()

    coh = create_attractors()
    ray.init()
    print('Number of agents is:{} and number of bits is: {}'.format(num_agents, number_of_bits))
    start = time.time()
    for exp in range(num_experiments):
        for alpha in alphas:
            for i in network_parameters:
                G = nx.watts_strogatz_graph(num_agents, watts_strogatz_graph_param, i, seed=0) # FIX THIS! change rewire parameters as from different starting, 1 means random graph as each node is going to rewired and no structure is saved
                agents, states = setup_environment(G,coh,bit_mat,alpha)
                run_simulation(end_simulation_time, agents, states)

    end = time.time()
    print((end-start)/60.0)

if __name__ == '__main__':
     doit()
