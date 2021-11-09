import numpy as np
import pandas as pd
import AgentClass
import const
import random
import networkx as nx
from config import num_agents,number_of_bits
from scipy import stats
from collections import defaultdict
import json

def get_tau_distr():
        lower = 0
        upper = 1
        mu = 0.5
        sigma = 0.1
        N = 1000

        samples = stats.truncnorm.rvs(
          (lower-mu)/sigma,(upper-mu)/sigma,loc=mu,scale=sigma,size=N)

        return samples


def create_environment():
    list_agents = []
    tau_distr = get_tau_distr()

    for i in range(num_agents):
        in_state = np.random.randint(2, size=number_of_bits).tolist()
        a = AgentClass.Agent(name='agent{}'.format(i), tau=random.choice(tau_distr), initial_state=in_state)
        list_agents.append(a)

    # create network
    G = nx.newman_watts_strogatz_graph(num_agents, 10, 0.5, seed= 0)
#    nx.draw(G, with_labels=True, font_weight='bold') # to draw agents
    df = nx.to_pandas_adjacency(G, dtype=int)

    tmp_edges = df.apply(lambda row: row.to_numpy().nonzero()).to_dict()
    edges = {k: v[0].tolist() for k, v in tmp_edges.items()}

    # make random connections with agents
    for k, v in edges.items():
        for ngh in v:
            list_agents[k].add_neighbors(list_agents[ngh])

    return list_agents

def get_network_df(list_agents):
    network_df = pd.DataFrame({'Agent Name':[], 'Neighbors':[]})
    for agt in list_agents:
        neighbors = agt.get_neighbors_name()
        network_df = network_df.append({'Agent Name':agt.name,
                                        'Neighbors':neighbors}, ignore_index=True)
    return network_df

def run_simulation(alpha, const:const.Constants, list_agents, end_time):
    d = []
    generations = 0
    for t in range(end_time):
        # compute next state for all agents
        for agt in list_agents:
            agt.update_knowledge(alpha, const.coh_transition_matrix, const.bit_matrix)

        # keep record of current record and all other values
        for agt in list_agents:
            row = {'Agent_Name':agt.name,
                   'Agent_Dissonance':agt.dissonance_lst,
                   'Time':t,
                   'Current_Knowledge_State':agt.knowledge_state,
                   'Next_Knowledge_State':agt.next_state}

            d.append(row)

        # now update all agents next state with computed next state
        for agt in list_agents:
            agt.knowledge_state = agt.next_state
            agt.next_state = None
            agt.dissonance_lst = None

        generations+=1
        if generations%10 == 0:
            print("alpha = {}; {} generations".format(alpha,generations))

    return pd.DataFrame(d)

if __name__ == '__main__':
    # constants intialization
    end_simulation_time = 200
    alphas = np.array(range(0,6))*.2
    exp_times = 1

    # first create environment
    agents_list = create_environment()

    # get network of the agents
    agent_network_df = get_network_df(agents_list)

    results = {}

    # for saving
    agent_network_df.to_json('test_network.json',orient='records', lines=True)

    for i in range(exp_times):

        random.seed(i)
        results['seed'] = i

        constants = const.Constants()

        results['coherence_matrix'] = coherence_matrix = constants.get_coh_tx_matrix().tolist()


        results['alphas'] = defaultdict(list)

        # run simulation
        for alpha in alphas:
#            record_df = pd.DataFrame({'Agent_Name':[], 'Agent_Dissonance':[], 'Time':[], 'Current_Knowledge_State':[], 'Next_Knowledge_State':[]})

            record_df = run_simulation(alpha, constants, agents_list, end_simulation_time)
            results['alphas'][alpha].append(record_df.to_json(orient='records', lines=True))



        with open('test_result.json', 'w') as fp:
            json.dump(results, fp, indent=4)
