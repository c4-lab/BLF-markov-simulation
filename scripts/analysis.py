import json

import pandas as pd
import numpy as np
import networkx as nx
import node2vec
import config
from gensim.models import Word2Vec
from scipy.special import softmax

import const
import random
import os, shutil

from scipy import stats
import utilities

from random import shuffle
import AgentClass


results_file = "test_result.json"
soc_network_file = "test_network.json"
coh_network_file = "coherence.edgelist"
embeddings_file = "_coherence.emb"
num_agents = 1024
number_of_bits = 10



def normalize_rows(m,noise=.001):
    for i, row in enumerate(m):
        for j, col in enumerate(row):
            if m[i][j] ==0:
                m[i][j] += noise

        summing = sum(row)
        for j, col in enumerate(row):
            m[i][j] = m[i][j]/summing

def hamming(a,b):
    return(bin(a^b).count("1"))

def init_coherence_matrix(nob,attractors,search_distance,inertial_weight=1,baseline = 1):
    attrctr_space_vec = np.zeros(2**nob)+baseline
    attractor_df = []
    for a in attractors:
        location = a[0]
        depth = a[1]
        radius = a[2]

        for j in range(2**nob):
            diff = hamming(j, location)
            attractor_df.append({"attractor":a,"position":j,"depth":depth,"distance":diff,"inbasin":diff<=radius})
            #diff = abs(j-k) ### Uncomment if needed to look into euclidean space
            if diff <= radius:
                attrctr_state_distance = (1-diff/radius)*depth
                attrctr_space_vec[j] = max(attrctr_space_vec[j], attrctr_state_distance)

    attrctr_space_mat = np.tile(attrctr_space_vec, (2**nob, 1))


    # create inertial matrix
    inertia_matrix = np.zeros((2**nob, 2**nob))

    for row_st, row in enumerate(inertia_matrix):
        for col_st, col in enumerate(row):
            bits_difference = hamming(row_st, col_st)
            #         bits_difference = np.abs(row_st- col_st)
            inertia_matrix[row_st, col_st] = max(0, 1- bits_difference *(1/search_distance))

    coherence_matrix = attrctr_space_mat
    if inertial_weight > 0:
         coherence_matrix*=(inertial_weight*inertia_matrix)
    normalize_rows(coherence_matrix,0)


    return attractor_df,coherence_matrix


def init_coherence_matrix_niraj(attractors, number_of_bits):
    
    """attractors is expected as dictionary in following format:
                {attrctr1: {'depth': 7, 'radius': 1},
                 attrctr2: {'depth': 9, 'radius': 4},
                 attrctr3: {'depth': 5, 'radius': 4}}"""
    
    attrctr_space_vec = np.zeros(2**number_of_bits)

    for k, v in attractors.items():
        attrctr_space_vec[k] = v['depth']
        r = v['radius']
        n = 2**number_of_bits
        for j in range(2**number_of_bits):
                diff = hamming(j, k)
    #             diff = abs(j-k) ### Uncomment if needed to look into euclidean space
                if diff <= r:
                    attrctr_state_distance = (1-diff/v['radius'])*v['depth']
                    attrctr_space_vec[j] = max(attrctr_space_vec[j], attrctr_state_distance)

    attrctr_space_mat = np.tile(attrctr_space_vec, (2**number_of_bits, 1))
        
    
    # create transition matrix
    inertia_matrix = np.zeros((2**number_of_bits, 2**number_of_bits))

    max_bits = 3 # maximum bits to for the transitions

    for row_st, row in enumerate(inertia_matrix):
        for col_st, col in enumerate(row):
            bits_difference = hamming(row_st, col_st)
            inertia_matrix[row_st, col_st] = (max_bits - min(max_bits, bits_difference))

          
    x = attrctr_space_mat*inertia_matrix # attractor space has twice impact than inertia
    
    minimum = x.min()
    # add noise to coherence matrix:
    for i, row in enumerate(x):
        for j, col in enumerate(row):
            if col == 0:
                x[i][j] += minimum

    coherence_matrix = softmax(x, axis=1)

    
    return attrctr_space_mat,inertia_matrix,coherence_matrix   

def dump_coherence_matrix_to_edgelist(write = True):
    with open(results_file) as f:
        results = json.load(f)
    adj_matrix = np.array(results['coherence_matrix'])
    G = nx.from_numpy_matrix(adj_matrix,create_using = nx.MultiDiGraph)
    if write:
        nx.write_edgelist(G,coh_network_file,data=['weight'])
    return G


def learn_embeddings(walks,dimensions):
    '''
    Learn embeddings by optimizing the Skipgram objective using SGD.
    '''
    walks = [list(map(str, walk)) for walk in walks]
    model = Word2Vec(walks, size=dimensions, window=config.n2v_window_size, min_count=0, sg=1, workers=config.n2v_workers, iter=config.n2v_iter)
    model.wv.save_word2vec_format(embeddings_file)

    return


def create_embedding(ndim=10,p = 1, q = 1,G = None):
    '''

    :param ndim:  Number of dimensions for the embedding
    :param p: Node2Vec 'return' hyperparameter
    :param q: Node2Vec 'input' hyperparmeter
    :param G: Graph for processing; will read from file if absent
    :return:
    '''
    if G is None:
        G = nx.read_edgelist(coh_network_file, nodetype=int, data=(('weight',float),), create_using=nx.DiGraph())
    G2 = node2vec.Graph(G, True, p, q)
    G2.preprocess_transition_probs()
    walks = G2.simulate_walks(config.n2v_num_walks, config.n2v_walk_length)
    learn_embeddings(walks,ndim)

def get_tau_distr():
    lower = 0
    upper = 1
    mu = 0.5
    sigma = 0.1
    N = num_agents

    samples = stats.truncnorm.rvs(
        (lower-mu)/sigma,(upper-mu)/sigma,loc=mu,scale=sigma,size=N)

    return samples

def create_even_bits_init_states():
    eq_bits = [[0 for i in range(num_agents//2)] + [1 for i in range(num_agents//2)] for i in range(number_of_bits)]
    for i in eq_bits:
        shuffle(i)
    return [[bit.pop() for bit in eq_bits] for i in range(num_agents)]


def binary_rep(n):
    fmt = '{0:0'+str(number_of_bits)+'b}'
    return [[int(i) for i in fmt.format(x)] for x in n]

def create_environment(inital_states,G=None):
    list_agents = []
    tau_distr = get_tau_distr()
    for i in range(num_agents):
        in_state = inital_states.pop()
        a = AgentClass.Agent(name='agent{}'.format(i), tau=random.choice(tau_distr), initial_state=in_state)
        list_agents.append(a)

    # create network
    if G is None:
        G = nx.barabasi_albert_graph(num_agents, 2, seed= 0)
    df = nx.to_pandas_adjacency(G, dtype=int)

    tmp_edges = df.apply(lambda row: row.to_numpy().nonzero()).to_dict()
    edges = {k: v[0].tolist() for k, v in tmp_edges.items()}

    # make random connections with agents
    for k, v in edges.items():
        for ngh in v:
            list_agents[k].add_neighbors(list_agents[ngh])

    return list_agents

def reinit_agents(agents,initial_states):
    copy = []+initial_states
    for a in agents:
        if len(initial_states)==0:
            print("Recycling state vector")
            initial_states = []+copy
        a.reset_state(initial_states.pop())

def get_network_df(list_agents):
    network_df = pd.DataFrame({'source':[], 'target':[]})
    for agt in list_agents:
        neighbors = agt.get_neighbors_name()
        for n in neighbors:
            network_df = network_df.append({'source':agt.name,
                                            'target':n}, ignore_index=True)
    return network_df

def run_simulation(alpha, coherence, bit_mat, list_agents, end_time, exp):
    d = []
    generations = 0
    for t in range(end_time):
        # compute next state for all agents
        for agt in list_agents:
            agt.update_knowledge(alpha, coherence, bit_mat)

        # keep record of current record and all other values
        for agt in list_agents:
            row = {'Agent_Number': int(agt.name.split('t')[1]),
                   'Time':t,
                   # at any time step we will need normalized how many neighbors disagree on bits
                   'bits_disagreement':np.mean(agt.state_disagreements),
                   #'Current_Knowledge_State':agt.knowledge_state,
                   'Current': utilities.bool2int(agt.knowledge_state),
                   'alpha':alpha,
                   'Next': utilities.bool2int(agt.next_state),
                   #'Next_Knowledge_State':agt.next_state,
                   "Experiment":exp}

            d.append(row)

        # now update all agents next state with computed next state
        for agt in list_agents:
            agt.knowledge_state = agt.next_state
            agt.next_state = None
            agt.dissonance_lst = None

        generations+=1

    return pd.DataFrame(d)


def create_graph_from_records(records:pd.DataFrame, remove_self_loops = True):
    records["weight"] = 1
    x=records[["Current","Next","weight"]].groupby(["Current","Next"]).agg('count').reset_index()
    if remove_self_loops:
        x.drop(x[x.Current == x.Next].index, inplace=True)
    xg = nx.from_pandas_edgelist(x,"Current","Next","weight",create_using = nx.MultiDiGraph)
    return xg


def path(sim,file,create=True):
    path = "../simulations/{}".format(sim)
    if create:
        if not os.path.isdir(path):
            os.mkdir(path)
    return "{}/{}".format(path,file)

def save_experiment(sim,config,attractor_df, coherence_matrix, inertial_weight, results_df:pd.DataFrame,network_df:pd.DataFrame):
    with open(path(sim,"config.txt"),"w") as f:
        f.write(str(config))
    pd.DataFrame(coherence_matrix).to_csv(path(sim,"coherence_matrix.csv"))
    pd.DataFrame(attractor_df).to_csv(path(sim,"attractors.csv"))
    with open(path(sim,"attractors.csv"),"a") as f:
        line = "# inertia {}".format(inertial_weight)
        f.write(line)
    results_df.to_csv(path(sim,"simdata.csv"))
    network_df.to_csv(path(sim,"network.csv"))

def produce_init_states(num):
    n = list(range(num))
    random.shuffle(n)
    return binary_rep(n)

def run_full_experiment(coherence_matrix,id,end_simulation_time = 100, exp_times = 5,reset_graph= False,alphas = None ):
    if alphas is None:
        alphas = [0,.2,.5,.7,.9]

    # first create environment
    agents_list = None
    constants = const.Constants()
    bit_mat = constants.get_bit_matrix()
    record_df = pd.DataFrame()
    for i in range(exp_times):
        print("Experiment {} or {}: {} agents for {} time steps".format(i,exp_times,num_agents,end_simulation_time))
        # run simulation
        for alpha in alphas:
            print(".",end="")
            init_state = produce_init_states(2**number_of_bits)
            if reset_graph or agents_list is None:
                agents_list = create_environment(init_state)
            else:
                reinit_agents(agents_list,init_state)
            tmp_record_df = run_simulation(alpha, coherence_matrix, bit_mat, agents_list, end_simulation_time,i)
            record_df = record_df.append(tmp_record_df)
    config = {"num_agents":num_agents,"num_bits":number_of_bits}
    save_experiment(id,config,record_df,get_network_df(agents_list),"simulation_data")

def ci2020_paper(replications = 10,timesteps = 100):
    alphas = np.log10(np.array(range(1,21)))/np.log10(20)
    G = nx.barabasi_albert_graph(num_agents, 2, seed= 0)
    agents_list = None
    glob_attractor = 5
    loc_attractor = list(range(10))
    loc_attractor.remove(glob_attractor)
    bit_mat = const.Constants().get_bit_matrix()

    for att in range(1,1+len(loc_attractor)):
        print("Cell {} of {}".format(att,len(loc_attractor)))
        attractors = [[2**x,100,1] for x in loc_attractor[0:att]]+[[2**glob_attractor,200,1]]
        adf,cm = init_coherence_matrix(number_of_bits,attractors,5)
        record_df = pd.DataFrame()
        for i in range(replications):
            print(".",end="")
            for alpha in alphas:
                init_state = produce_init_states(2**number_of_bits)
                if agents_list is None:
                    agents_list = create_environment(init_state,G)
                else:
                    reinit_agents(agents_list,init_state)
                tmp_record_df = run_simulation(alpha,cm, bit_mat, agents_list, timesteps,i)
                record_df = record_df.append(tmp_record_df)
        config = {"num_agents":num_agents,"num_bits":number_of_bits}
        save_experiment("final_ci2020_{}".format(att),config,adf,cm,1,record_df,get_network_df(agents_list))
        print("Done cell.")

def extract_communities_directed(G:nx.DiGraph):
    pass

