import networkx as nx
import numpy as np
from blf.utilities import hamming, normalize_rows, hamming_neighbors, truncnorm, get_global_attractors, scale

import random
from sklearn.preprocessing import normalize
import functools
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

def create_attractors_from_energy_surface(surface,width=0):
    states = [x[1] for x in surface]
    mn, mx = min(states),max(states)
    f = lambda x: scale(x,mn,mx)
    return [(x[0],f(x[1]),width) for x in surface]


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




def build_hamming_distance_matrix(bits):
    inertia_matrix = np.zeros((2**bits,2**bits))
    for row_st, row in enumerate(inertia_matrix):
        for col_st, col in enumerate(row):
            bits_difference = hamming(row_st, col_st)
            inertia_matrix[row_st, col_st] = (bits- bits_difference)/bits
    return inertia_matrix

def build_hamming_mask(bits, width):
    inertia_matrix = np.zeros((2**bits,2**bits))
    for row_st, row in enumerate(inertia_matrix):
        for col_st, col in enumerate(row):
            bits_difference = hamming(row_st, col_st)
            if bits_difference > width:
                continue
            inertia_matrix[row_st, col_st] = 1
    return inertia_matrix

# def modify_search_landscape(attractor_matrix):
#     diag = attractor_matrix.diagonal()
#     best_val = max(diag)
#     global_attractors = [i for i, v in enumerate(diag) if v == best_val]
#     gradient = []
#     nbits = int(math.log(len(attractor_matrix),2))
#     for g in global_attractors:
#         gradient.append(functools.reduce(
#             lambda x,y,g=g,w=nbits: x + hamming_neighbors(g,w,y),
#             range(1,nbits+1),
#             []
#         ))
#     visited = set()
#     for i in range(len(gradient[0])):
#         for j in len(gradient):
#             state = gradient[j].pop(0)
#             if state not in visited:
#                 visited.add(state)
#
#



# Number is -w**2 to w**2; negative numbers indicate a simplification of the space
def modify_landscape(a,w,number = None):
    landscape = dict(a)
    missing = set(range(w**2)) - landscape.keys()
    if len(missing):
        landscape.update(dict(zip(list(missing),[0]*len(missing))))
    maxval = max(landscape.values())
    glbls = [x[0] for x in filter(
        lambda x: x[1]==maxval,a)]

    if number is None:
        number = w**2
    if number < 0:
        sorted_attractors = sorted(landscape.items(),key = lambda x: x[1],reverse=True)
        number = -number
    else:
        sorted_attractors = sorted(landscape.items(),key = lambda x: x[1])

    updated = {g: landscape.pop(g) for g in glbls}
    gradient = []
    for g in glbls:
        gradient.append(functools.reduce(
            lambda x,y,g=g,w=w: x + hamming_neighbors(g,w,y),
            range(1,w+1),
            []
        ))

    #Not very pythonic!
    for i in range(min(number,len(gradient[0]))):
        for j in range(len(glbls)):
            if gradient[j][i] in landscape:
                old_weight = landscape.pop(gradient[j][i])
                old_attractor = sorted_attractors.pop(0)
                updated[gradient[j][i]] = old_attractor[1]
                # Uncertain if this is necessary?
                if old_attractor[0] in landscape:
                    landscape[old_attractor[0]] = old_weight


    return list({**landscape,**updated}.items())

def empower(m,factor = 1):
    result = np.power(factor,m)
    return normalize(result,norm="l1")

def amplify(m,factor = 1):
    result = np.power(m,factor)
    return normalize(result,norm="l1")


def generate_random_attractors(bits, mu = .5, sigma=.15, num_globals=1, mode ="normal"):
    all_attractors = list(range(2**bits))
    random.shuffle(all_attractors)
    if mode == "normal":
        weights =truncnorm(size=2 ** 8 + 1 - num_globals, mu=mu, sigma=sigma)
    elif mode == "uniform":
        weights = np.random.uniform(0, 1, size=2 ** 8 + 1 - num_globals)
    attractors = list(zip(all_attractors[:len(weights)],weights))
    if num_globals > 1:
        attractors+=list(zip(all_attractors[len(weights):], [max(weights)] * (num_globals - 1)))
    return attractors

def generate_tuned_surface(bits, attractors):
    diag = np.zeros(2**bits)
    attractors = sorted(attractors,key = lambda x: x[1])
    for a in attractors:
        diag[a[0]] = a[1]
        max_width = a[2]
        delta = a[1] / max_width
        start = a[1] - delta
        for w in range(1,max_width+1):
            neighbors = hamming_neighbors(a[0],8,w)
            for n in neighbors:
                diag[n] = max(diag[n],start / len(neighbors))
            start = max(0,start - delta)
    attractor_list = [(i,x) for i,x in enumerate(diag)]
    return build_attractor_profile(bits,attractor_list)

def build_scaled_hamming_distance_matrix(bits, width):
    inertia_matrix = np.zeros((2**bits,2**bits))
    for row_st, row in enumerate(inertia_matrix):
        start = 1
        delta = start / width
        for dist in range(width+1):
            neighbors = hamming_neighbors(row_st,bits,dist)
            for n in neighbors:
                inertia_matrix[row_st][n] = start / len(neighbors)
            start = start - delta
    return inertia_matrix

def build_attractor_profile(bits,attractors):
    arow = np.ones((2**bits)) * 1/bits

    # Presume here that attractors vary from -1 to 1; this function just places the zero crossing at 1/# bits
    f = lambda x: ((x+1)/2)**math.log(1/bits,.5)
    attractors = sorted(attractors,key = lambda x: x[1],reverse=True)
    attractors = [(x[0],f(x[1]),x[2]) for x in attractors]
    global_best, global_worst = attractors[0][1], attractors[-1][1]
    pre_gen_globals = set([i for i,j,k in attractors if j==global_best])

    #Set down attractors

    for a in attractors:
        #arow[a[0]] = a[1]
        max_d = a[2]+1
        for w in range(0,max_d):
            neighbors = hamming_neighbors(a[0],8,w)
            #print(f"Neighbors of {a[0]} at dist {w}: {neighbors}")
            for n in neighbors:
                # Here the exponent (hard-coded at 2) determines the slope of the gradient
                weight = a[1]*((max_d-w)/max_d)**2
                arow[n] = max(weight,arow[n])

    result = np.tile(arow,(2**bits,1))
    post_gen_globals = set([i for i,j in get_global_attractors(result)])
    if pre_gen_globals != post_gen_globals:
        print(f"WARNING: requested globals {pre_gen_globals} != generated globals {post_gen_globals}")
    return result


def build_manual_transition_matrix(bits,raw_attractor_profile,search_width = -1, stickiness = 1,att_str = 3):
    if search_width <0 or search_width > bits:
        search_width = int(bits / 2)
    hm = build_scaled_hamming_distance_matrix(8,search_width)
    ap = amplify(raw_attractor_profile,att_str)
    return(amplify(ap*amplify(hm,stickiness)))


def buildRandomIsingBasedTransitionMatrix(bits,pzero):
    net = buildRandomNetwork(bits,pzero)
    matrix = buildIsingAttractorMatrix(net)
    return matrix

# def build_manual_transition_matrix(bits, attractors):
#     attrctr_space_vec = np.zeros(2 ** bits)
#     for a in attractors:
#         attrctr_space_vec[a[0]]=a[1]
#     result = normalize(np.tile(attrctr_space_vec,(2**bits,1)),norm="l1")
#     return result



def buildIsingMatrixWithLocalSearch(net,noise = .05):
    bits = net.number_of_nodes()
    m = amplify(buildIsingAttractorMatrix(net)+noise,1)
    h = amplify(build_hamming_distance_matrix(bits), bits / 2)
    mh1 = m * h * build_hamming_mask(bits, bits / 2)
    return amplify(mh1,2)

def trial1ManualMatrix(bits,attractors):
    #print("hi")
    m = amplify(build_manual_transition_matrix(bits,attractors)+.05,1)
    h = amplify(build_hamming_distance_matrix(bits), bits / 2)
    mh1 = m * h * build_hamming_mask(bits, bits / 2)
    return amplify(mh1,2)

def trial1IsingMatrix(bits,pzero = .5,noise = .05):
    net = buildRandomNetwork(bits,pzero)
    return buildIsingMatrixWithLocalSearch(net,noise)
    # m = amplify(buildRandomIsingBasedTransitionMatrix(bits,pzero)+noise,1)
    # h = amplify(buildHammingDistanceMatrix(bits),4)
    # mh1 = m*h*buildHammingMask(bits,bits/2)
    # return amplify(mh1,2)


def localize_transition_matrix(m,noise = .05):
    bits = int(math.log(m.shape[0],2))
    m = amplify(m + noise,1)
    h = amplify(build_hamming_distance_matrix(bits), 1)
    mask = build_hamming_mask(bits, bits / 2)
    return amplify(m*h*mask,2)

# Default settings give a nice even distribution across a range from -1 to 1 for and unchanged attractor matrix
# For a localized matrix, recoomend going 0 to 1, with slope = .3 and period = 2.6
def generate_manual_landscape_sweep(bits,replications,linspace, remap = True, slope = .2, period = 2.1):
    f = lambda x: slope * math.tan((math.pi * x)/period)
    data = []
    for j in linspace:
        if remap:
            deviation = int(f(j)* (2**bits))
        else:
            deviation = int(j * 2**bits)
        for i in range(replications):
            start = generate_random_attractors(bits,mode="uniform")
            data.append(modify_landscape(start,8,deviation))
    return data




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
