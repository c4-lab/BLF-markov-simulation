import networkx as nx
import pandas as pd
import numpy as np
from blf.utilities import hamming, normalize_rows
from random import random as rnd
from sklearn.preprocessing import normalize
import math


# Ising model

def buildNetwork(num):
    mat = np.zeros((num,num))
    for x in range(num):
        for y in range(num):
            if x == y:
                continue
            dist = min((x-y)%num,(y-x)%num)
            if dist%num == 1:
                mat[x,y]=1
            else:
                mat[x,y]=-1
            #elif dist%num <=2:
            #    mat[x,y]=-1
    return nx.from_numpy_matrix(mat)

def buildRandomNetwork(num,pzero = .1):
    mat = np.zeros((num,num))
    for x in range(num):
        for y in range(x):
            if np.random.random()<pzero:
                continue
            mat[x,y] = np.random.random()*2 - 1
            #elif dist%num <=2:
            #    mat[x,y]=-1
    return nx.from_numpy_matrix(mat)

def calcEnergySurface(net):
    nodes = len(net.nodes())
    result = []
    for x in range(2**nodes):
        s = [i if i==1 else -1 for i in [int(i) for i in np.binary_repr(x,nodes)]]
        s.reverse()
        weight = -sum([s[u]* s[v]* edata['weight']  for u,v,edata in net.edges(data=True)])
        result.append((x,weight))
    return result

def collectEnergyStates(net):
    result = calcEnergySurface(net)
    hist = {}
    for i in result:
        if i[1] in hist:
            hist[i[1]].append(i[0])
        else:
            hist[i[1]] = [i[0]]
    return hist


def buildIsingAttractorMatrix(net):
    weight = dict(calcEnergySurface(net))
    mx,mn = max(weight.values()),min(weight.values())
    # Remembering here that we're looking for *low* energy states, hence the subtract from 1
    scaled = [1-(weight[w]-mn)/(mx-mn) for w in range(len(weight))]
    mat = [scaled] * len(scaled)
    return normalize(np.array(mat),norm="l1")

def buildHammingDistanceMatrix(bits):
    inertia_matrix = np.zeros((2**bits,2**bits))
    for row_st, row in enumerate(inertia_matrix):
        for col_st, col in enumerate(row):
            bits_difference = hamming(row_st, col_st)
            inertia_matrix[row_st, col_st] = (bits- bits_difference)/bits
    return inertia_matrix

def buildHammingMask(bits,width):
    inertia_matrix = np.zeros((2**bits,2**bits))
    for row_st, row in enumerate(inertia_matrix):
        for col_st, col in enumerate(row):
            bits_difference = hamming(row_st, col_st)
            if bits_difference > width:
                continue
            inertia_matrix[row_st, col_st] = 1
    return inertia_matrix



def empower(m,factor = 1):
    result = np.power(factor,m)
    return normalize(result,norm="l1")

def amplify(m,factor = 1):
    result = np.power(m,factor)
    return normalize(result,norm="l1")


def buildRandomIsingBasedTransitionMatrix(bits,pzero):
    net = buildRandomNetwork(bits,pzero)
    matrix = buildIsingAttractorMatrix(net)
    return matrix

def buildManualTransitionMatrix(bits, attractors):
    attrctr_space_vec = np.zeros(2 ** bits)
    for a in attractors:
        attrctr_space_vec[a[0]]=a[1]
    result = normalize(np.tile(attrctr_space_vec,(2**bits,1)),norm="l1")
    return result

def trial1ManualMatrix(bits,attractors):
    m = amplify(buildManualTransitionMatrix(bits,attractors)+.05,1)
    h = amplify(buildHammingDistanceMatrix(bits),bits/2)
    mh1 = m*h*buildHammingMask(bits,bits/2)
    return amplify(mh1,2)

def trial1IsingMatrix(bits,pzero = .5,noise = .05):
    m = amplify(buildRandomIsingBasedTransitionMatrix(bits,pzero)+noise,1)
    h = amplify(buildHammingDistanceMatrix(bits),4)
    mh1 = m*h*buildHammingMask(bits,bits/2)
    return amplify(mh1,2)



# Manual construction

# For manual configuration, we would pass something like this into the 'attractors' parameter

# attractors_dict_lst = [{'state':32, 'radius':1, 'depth':200}, {'state':2, 'radius':1, 'depth':100}, {'state':4, 'radius':1, 'depth':100},
#                        {'state':8, 'depth':100, 'radius':1},{'state':16, 'radius':1, 'depth':100}, {'state':64, 'depth':100, 'radius':1},
#                        {'state':128, 'radius':1, 'depth':100}, {'state':256, 'radius':1, 'depth':100}, {'state':512, 'depth':100, 'radius':1},
#                        {'state':1024, 'radius':1, 'depth':100}]


def buildManualCoherenceMatrix_old(bits, attractors, search_distance, inertial_weight=1, baseline = 1):
    attrctr_space_vec = np.zeros(2 ** bits) + baseline
    attractor_df = []
    for a in attractors:
        location = a[0]
        depth = a[1]
        radius = a[2]

        for j in range(2 ** bits):
            diff = hamming(j, location)
            attractor_df.append({"attractor":a,"position":j,"depth":depth,"distance":diff,"inbasin":diff<=radius})
            #diff = abs(j-k) ### Uncomment if needed to look into euclidean space
            if diff <= radius:
                attrctr_state_distance = (1-diff/radius)*depth
                attrctr_space_vec[j] = max(attrctr_space_vec[j], attrctr_state_distance)

    attrctr_space_mat = np.tile(attrctr_space_vec, (2 ** bits, 1))


    # create inertial matrix
    inertia_matrix = np.zeros((2 ** bits, 2 ** bits))

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
