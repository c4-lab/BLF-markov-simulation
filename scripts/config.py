# constants for creating coherence matrix 
number_of_bits = 4

seed_val = [0]
coh_mat_min_value = 0.5
coh_mat_max_value = 0.9

#Probably not useful now
coh_mat_max_bits_to_flip = number_of_bits
coh_mat_min_bits_to_flip = 0

#Used to generate coherence transition matrix
alpha_d = .03 # Symmetric dirichlet parameter for generating coherence matrix
cliff = True   # Should we obey a min probability?
min_prob = .001  # Min probability if cliff is True




# for Agents class
contagion_mode='viral'
scale = 0.8

# other parameters
num_agents = 50

# for node2vec analysis
n2v_window_size = 10
n2v_workers = 8
n2v_iter = 1
n2v_num_walks = 10
n2v_walk_length = 80
