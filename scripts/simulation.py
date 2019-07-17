import numpy as np
import pandas as pd
import AgentClass
import const
import random

def create_environment(knowledge_space_bits, number_of_agents, tau_distr, list_agents):
    for i in range(number_of_agents):
        in_state = np.random.randint(2, size=knowledge_space_bits).tolist()
        a = AgentClass.Agent(name='agent{}'.format(i), tau=random.choice(tau_distr), initial_state=in_state)
        list_agents.append(a)
        
    # make random connections with agents
    for i in range(10):
        sample_agents = random.sample(list_agents, 2) 
        if sample_agents[0].name not in sample_agents[1].get_neighbors():
            sample_agents[0].add_neighbors(sample_agents[1])

def get_network_df(list_agents):
    network_df = pd.DataFrame({'Agent Name':[], 'Neighbors':[]})
    for agt in list_agents:
        neighbors = agt.get_neighbors_name()
        network_df = network_df.append({'Agent Name':agt.name, 
                                        'Neighbors':neighbors}, ignore_index=True)
    return network_df
 
def run_simulation(alpha, coh_matrix, record_df, list_agents, end_time):
    for t in range(end_time):    
        # compute next state for all agents
        for agt in list_agents:
            agt.update_knowledge(alpha, coh_matrix) 
         
        # keep record of current record and all other values
        for agt in list_agents:
            row = {'Agent_Name':agt.name,
                   'Agent_Dissonance':agt.dissonance_lst,
                   'Time':t,
                   'Current_Knowledge_State':agt.knowledge_state,
                   'Next_Knowledge_State':agt.next_state}
            
            record_df = record_df.append(row, ignore_index=True)
        
        # now update all agents next state with computed next state
        for agt in list_agents:
            agt.knowledge_state = agt.next_state
            agt.next_state = None
            agt.dissonance_lst = None
            
    return record_df

if __name__ == '__main__':
    # constants intialization
    agents_list = []
    knowledge_bits = 5
    num_agents = 5
    end_simulation_time = 5
    alpha = 0.5
    
    constants = const.Constants(knowledge_bits)
    tau_distribution = constants.get_tau_distr()
    coherence_matrix = constants.get_coh_matrix() 
    record_df = pd.DataFrame({'Agent_Name':[], 'Agent_Dissonance':[], 'Time':[], 'Current_Knowledge_State':[], 'Next_Knowledge_State':[]})

    # first create environment
    create_environment(knowledge_bits, num_agents, tau_distribution, agents_list)
    
    # get network of the agents
    agent_network_df = get_network_df(agents_list) 
    # run simulation
    record_df = run_simulation(alpha, coherence_matrix, record_df, agents_list, end_simulation_time)
    
    coh_matrix_df = pd.DataFrame(coherence_matrix)
    
    agent_network_df.to_json('test_network.json',orient='records', lines=True)
    record_df.to_json('test_simulation.json',orient='records', lines=True)
    agent_network_df.to_csv('test_network.csv', index=False)
    record_df.to_csv('test_simulation.csv', index=False)
    coh_matrix_df.to_csv('test_coherence_matrix.csv', index=False)

