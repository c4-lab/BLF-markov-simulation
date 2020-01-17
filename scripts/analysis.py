import json

import pandas as pd
import numpy as np
import networkx as nx
import node2vec
import config
from gensim.models import Word2Vec

results_file = "test_result.json"
soc_network_file = "test_network.json"
coh_network_file = "coherence.edgelist"
embeddings_file = "_coherence.emb"


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