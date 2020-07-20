import numpy as np

# constants for creating coherence matrix
number_of_bits = 10 # setting for number of bits
num_agents = 1024   # setting for number of agents

# attractors description

attractors_dict_lst = [{'state':2, 'radius':1, 'depth':100}, {'state':4, 'radius':1, 'depth':100}, {'state':8, 'depth':100, 'radius':1},
{'state':16, 'radius':1, 'depth':100}, {'state':32, 'radius':1, 'depth':200}, {'state':64, 'depth':100, 'radius':1},
{'state':128, 'radius':1, 'depth':100}, {'state':256, 'radius':1, 'depth':100}, {'state':512, 'depth':100, 'radius':1},
{'state':1024, 'radius':1, 'depth':100}]

# attractors radius and depth parameters 
attrctr_min_depth = 1
attrctr_max_depth = 5

attrctr_min_radius = 1
attrctr_max_radius = 5

# parameters for tau distribution of agents
tau_lower_bound = 0.1
tau_upper_bound = 1
tau_mu = 0.5
tau_sigma = 0.1
tau_n_samples = 1000


# pool_param = 1

# network x parameter
watts_strogatz_graph_param = 10

sim_network_params_lst = np.linspace(0, 1, 11)
end_sim_time = 100
alpha_range = np.linspace(0, 1, 11)

num_experiments = 10

seed_val = [0]
coh_mat_min_value = 0.5
coh_mat_max_value = 0.9

#Probably not useful now
coh_mat_max_bits_to_flip = number_of_bits
coh_mat_min_bits_to_flip = 0

#Used to generate coherence transition matrix

# Method depends on implementation - currently one of: dirichlet, bitwise
tx_init_method = "bitblock"
cliff = True   # Should we obey a min probability?
min_prob = .001  # Min probability if cliff is True

#method specific params

#dirichlet
alpha_d = .9 # Symmetric dirichlet parameter for generating coherence matrix

#bitwise
max_bits_to_flip = number_of_bits
min_bits_to_flip = 0
flip_min_value = .5
flip_max_value = 1


#fixedattractor
#attractors = [(0,0),(682,.2),(1023,.3)]
#fa_max_dist = 3
#fa_max_prob = .9




# for Agents class
contagion_mode='viral'
scale = 0.8

# for node2vec analysis
n2v_window_size = 10
n2v_workers = 8
n2v_iter = 1
n2v_num_walks = 40
n2v_walk_length = 80


# for tau
#tau_lower_bound = 0
#tau_upper_bound = 1
#tau_mu =  .5
#tau_sigma = 0.1
#tau_n_samples = num_agents
