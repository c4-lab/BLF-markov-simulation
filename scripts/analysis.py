import json

import pandas as pd
import numpy as np
import networkx as nx
import node2vec
import config
from gensim.models import Word2Vec
from scipy.special import softmax

results_file = "test_result.json"
soc_network_file = "test_network.json"
coh_network_file = "coherence.edgelist"
embeddings_file = "_coherence.emb"



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

    pd.DataFrame(attractor_df).to_csv("attractors.csv")
    with open("attractors.csv","a") as f:
        line = "# inertia {}".format(inertial_weight)
        f.write(line)

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
    pd.DataFrame(coherence_matrix).to_csv("coherence_matrix.csv")

    return attrctr_space_mat,inertia_matrix,coherence_matrix


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


def extract_communities_directed(G:nx.DiGraph):
    pass