# constants for creating coherence matrix 
number_of_bits = 4

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
attractors = [(0,0),(682,.2),(1023,.3)]
fa_max_dist = 3
fa_max_prob = .9




# for Agents class
contagion_mode='viral'
scale = 0.8

# other parameters
num_agents = 200

# for node2vec analysis
n2v_window_size = 10
n2v_workers = 8
n2v_iter = 1
n2v_num_walks = 40
n2v_walk_length = 80
