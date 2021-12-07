# This is a purely abstraction free version of the ABM
# commenting line below still works fine
#from __future__ import annotations
import numpy as np
import ray
import const
import random
import networkx as nx
# all parameters imported from the config file
from blf import config
#from config import number_of_bits, num_agents, tau_lower_bound, tau_upper_bound, tau_mu, tau_sigma, tau_n_samples, watts_strogatz_graph_param,sim_network_params_lst, end_sim_time, alpha_range, num_experiments, attrctr_min_depth, attrctr_max_depth, attrctr_min_radius, attrctr_max_radius, attractors_dict_lst
from scipy import stats
import analysis
import time
from blf import utilities
import os
import argparse
import re


import pandas as pd
import transition_matrices
from collections import defaultdict

shrd_static = {
    "coherence_matrix":None,
    "bit_matrix":None,
    "alpha":None
}

# def hamming(a,b):
#     return(bin(a^b).count("1"))

exp_number = 0


global constants



"""Each agent has coherence matrix and neighbors which are agents as well"""

class Agent:

    def __init__(self, idx, number_of_bits, tau_fx, tau=-1, alpha = .5):
        self.idx = idx
        self.neighbors = set() # neighbors are a list a of indices of other agents
        self.tau = tau # agent's threshold defined in initialization
        self.number_of_bits = number_of_bits
        self.alpha = alpha
        self.tau_fx = tau_fx

    def add_neighbor_indices(self, neighbors):
        """add neigbhor agent to the list of neighbor indices"""

        self.neighbors.update(neighbors)

    def update(self,static,dynamic):
        """Takes environment with common values to compute"""

        # first convert state binary to int to get the row in coherence matrix
        txn = static["coherence_matrix"]
        bit_matrix = static["bit_matrix"]


        #alpha = static["alpha"]

        kstate = dynamic[self.idx]

        # for agt in population:
        row_ptr = utilities.bool2int(kstate)

        # get the corresponding probabilites from the matrix
        coh_prob_tx = txn[row_ptr]
        ones_list = np.zeros(self.number_of_bits)

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


            dissonance = self.tau_fx(dissonance,self.tau)


            # transition probabilities given social pressure for moving to a state
            # with a '1' at this bit
            ones_list[kbit] = (1-dissonance if curr_bit_state else dissonance)

        zeros_list = 1-ones_list
        tmp_soc_mat = ones_list * bit_matrix  + zeros_list * (1-bit_matrix)

        # Probabilities for each state given social pressure
        soc_prob_tx = np.prod(tmp_soc_mat,1)
        #print(f"{soc_prob_tx} sum = {np.sum(soc_prob_tx)}")
        #TODO logs soc_prob_tx for each agent at each time step

        probs = self.alpha * soc_prob_tx + (1-self.alpha)*coh_prob_tx
        return utilities.int2bool(np.random.choice(range(2**self.number_of_bits),1,p=probs)[0],self.number_of_bits)



def init_agents(network:nx.Graph,tau_fx, tau, number_of_bits,alpha):
    list_agents = []

    for i in range(network.number_of_nodes()):
        a = Agent(i,number_of_bits,tau_fx,tau[i],alpha[i])
        list_agents.append(a)
        n = [x for x in network.neighbors(i)]
        a.add_neighbor_indices(n)

    # create network
    return list_agents


def setup_environment(config:config.Config, bit_mat):
    tau_type = config.get_round_settings("tau")["type"]

    if tau_type=="direct-proportion":
        def tau_fx(dissonance,tau):
            return dissonance
    elif tau_type == "sigmoid":
        def tau_fx(dissonance,tau):
            return utilities.sigmoid(dissonance, tau)

    list_agents = init_agents(config.graph,tau_fx,config.tau,config.number_of_bits,config.alpha)
    shrd_static["coherence_matrix"] = config.tx_matrix
    shrd_static["bit_matrix"] = bit_mat
    shrd_static["alpha"] = config.alpha

    values = list(range(config.tx_matrix.shape[0]))
    random.shuffle(values)
    states = []
    for i in range(config.number_of_agents):
        states.append(utilities.int2bool(values[i % len(values)],config.number_of_bits))

    return list_agents, states


@ray.remote
def agent_update(agents, static, dynamic):
    return [agent.update(static,dynamic) for agent in agents]


def run_simulation(end_time, agents, states):
    static_obj = ray.put(shrd_static)

    sim_result_lst = []

    ncores = os.cpu_count()


    for t in range(end_time):
        dynamic_obj = ray.put(states)
        chunked = chunks(agents,ncores)
        sim_result = defaultdict(list)
        for agt_index, agt_state in enumerate(states):
            sim_result['Agent_Number'].append(agt_index)
            sim_result['Time'].append(t)
            sim_result['Current_Knowledge_State'].append(utilities.bool2int(agt_state))

        results = [agent_update.remote(chunk,static_obj,dynamic_obj) for chunk in chunked]
        states = [item for sublist in ray.get(results) for item in sublist]
        for agt_nxt_state in states:
            sim_result['Next_Knowledge_State'].append(utilities.bool2int(agt_nxt_state))
        sim_result_lst.append(sim_result)
        del dynamic_obj
    return sim_result_lst

def chunks(lst, n):
    """Yield n chunks from lst."""
    result = [[] for i in range(n)]
    for i, agent in enumerate(lst):
        result[i % n].append(agent)
    return result



def runExperiment(config: config.Config, stub, outdir):
    detailed_results = []
    final_sim_results = []
    sim_parameters = []

    constants = const.Constants(config)
    bit_mat = constants.get_bit_matrix()
    ray.init()
    print('Running experiments ............ ')
    start = time.time()
    stub = f"{stub}-{time.strftime('%m%d%Y_%H%M%S', time.gmtime())}"

    while config.step():
        config.inspect()
        agents, states = setup_environment(config,bit_mat)
        #agents, states = setup_environment(config,config.graph,config.tx_matrix,bit_mat,config.alpha,config.number_of_agents,config.number_of_bits,config.tau)
        random.shuffle(states) # randomly shuffling states for each replication experiment number

        #  0 - non-global states, 1 - energy, 2 - globals, 3 - correlation (R,pval)
        surface_inspection = utilities.measure_landscape_complexity(config.tx_matrix)
        print(f"Landscape complexity = {surface_inspection[3][0]}")
        simulation_results = run_simulation(config.number_of_steps, agents, states)
        sim_df = pd.DataFrame(simulation_results)
        sim_df_exp = sim_df.apply(pd.Series.explode).reset_index()
        sim_df_exp.drop('index', axis=1, inplace=True)
        utilities.write_zip(sim_df_exp,f"{outdir}/{stub}-detailed",f'{stub}.{config.get_run_id()}.detail')
        utilities.write_zip(pd.DataFrame(config.tx_matrix),f"{outdir}/{stub}-detailed",f'{stub}.{config.get_run_id()}.tx_matrix')

        last_run = sim_df_exp[sim_df_exp["Time"]==config.number_of_steps-1]
        last_run = last_run.groupby(['Next_Knowledge_State']).agg({"Next_Knowledge_State":["count"]})
        last_run = last_run.reset_index()
        last_run.columns = ["state","count"]
        last_run["match"]=last_run.apply(lambda x: utilities.closest_state(x['state'],surface_inspection[2]), axis=1)
        last_run[['closest', 'distance']] = pd.DataFrame(last_run['match'].tolist(), index=last_run.index)
        last_run = last_run[['state','count','closest','distance']]
        last_run["run_id"] = config.get_run_id()

        final_sim_results.append(last_run)



        run_params = config.collect_parameters()
        run_params["run_id"] = config.get_run_id()
        run_params["complexity"] = surface_inspection[3][0]
        run_params["globals"] = ";".join([str(x) for x in surface_inspection[2]])
        sim_parameters.append(run_params)

    utilities.write_zip(pd.concat(final_sim_results),outdir,f"{stub}.summary")
    utilities.write_zip(pd.DataFrame.from_dict(sim_parameters),outdir,f'{stub}.params')

    end = time.time()
    print('> Experiment completed in {} minutes.'.format((end-start)/60.0))


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("config_file")
    parser.add_argument("output_dir")
    args = parser.parse_args()
    c = config.Config()
    c.process_config_file(args.config_file)
    stub = re.search('(?:/)?([^/]+?)(?:\.[^.]+)?$',args.config_file)[1]
    runExperiment(c,stub,args.output_dir)







