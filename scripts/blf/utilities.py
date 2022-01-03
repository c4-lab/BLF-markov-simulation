import math
import numpy as np
import scipy.stats as stats
import os
import itertools, functools

#from scipy.stats import pearsonr

def bool2int(x):
    """function for binary to decimal
    from: # https://stackoverflow.com/a/15505648/5916727"""
    y = 0
    for i,j in enumerate(x):
        y += j<<i
    return y

def truncnorm(size, low=0,high=1,mu = .5,sigma = .5):

    X = stats.truncnorm((low - mu) / sigma, (high - mu) / sigma, loc=mu, scale=sigma)
    return X.rvs(size)

def int2bool(x,width = None):
    """function for decimal to binary"""
    if not width:
        return [int(x) for x in reversed(list(np.binary_repr(x)))]
    else:
        return [int(x) for x in reversed(list(np.binary_repr(x,width)))]

def sigmoid(x, x_0, k=8):
    # https://en.wikipedia.org/wiki/Logistic_function
    return 1 / (1 + math.exp(-k*(x-x_0)))

def scale(x,fmin,fmax,tmin = -1,tmax=1):
    return (x-fmin)/(fmax-fmin) * (tmax-tmin) + tmin

def hamming(a,b):
    return(bin(a^b).count("1"))

def closest_state(probe,targets:list):
    val = min(targets,key=lambda x: hamming(probe,x))
    return (val,hamming(val,probe))

def hamming_neighbors(state,width,dist):
    result = []
    for i in itertools.combinations(range(width),dist):
        mask = functools.reduce(lambda x,y: x+(1<<y), i,0)
        result.append(state^mask)
    return result

def build_tan_mapped_linspace(low,high,count,slope = .3, period = 2.6):
    f = lambda x: slope * math.tan((math.pi * x)/period)
    return [f(x) for x in np.linspace(low,high,count)]

def get_global_attractors(attractor_matrix):
    diag = attractor_matrix.diagonal()
    best_val = max(diag)
    global_attractors = [(i,v) for i, v in enumerate(diag) if v == best_val]
    return global_attractors

def search_complexity(attractor_matrix):
    diag = attractor_matrix.diagonal()
    best_val = max(diag)
    global_attractors = [i for i, v in enumerate(diag) if v == best_val]
    total = 0
    for f,row in enumerate(attractor_matrix):
        curr_d = {g:hamming(f,g) for g in global_attractors}
        weights = {k:0 for k in global_attractors}
        for t,val in enumerate(row):
            for g in global_attractors:
                if hamming(t,g) < curr_d[g] or g==t:
                    weights[g] = weights[g] + val
        total += max(weights.values())
    return 1-(total / len(diag))

def empirical_complexity(attractor_matrix,iterations=10,maxpath=100):
    diag = attractor_matrix.diagonal()
    best_val = max(diag)
    global_attractors = [i for i, v in enumerate(diag) if v == best_val]
    choices = list(range(len(attractor_matrix)))
    results = np.zeros(len(attractor_matrix))
    for i,row in enumerate(attractor_matrix):
        for it in range(iterations):
            steps = 0
            next_row = row
            while(steps < maxpath):
                n = np.random.choice(choices,p=next_row)
                if n in global_attractors:
                    break
                else:
                    next_row = attractor_matrix[n]
                    steps+=1
            expected = 1 if i==n else hamming(i,n)
            took = steps+1
            trial = (took - expected) / expected
            results[i] = results[i] + (sigmoid(trial,0,1) - .5)*2
    results = results/iterations
    return np.mean(results)




def measure_landscape_complexity(attractor_matrix):
    # Jones & Forrest 1995
    diag = attractor_matrix.diagonal()
    best_val = max(diag)
    global_attractors = [i for i, v in enumerate(diag) if v == best_val]

    f = []
    d = []
    for i,j in enumerate(diag):
        if j == best_val:
            continue
        f.append(j)
        d.append(min([hamming(i,g) for g in global_attractors]))
    # Positive correlation indicates that the gradient travels away
    # from the attractors
    r = stats.pearsonr(f,d)

    complexity = (r[0] + 1) / 2
    return f,d,global_attractors,complexity


def my_softmax(x,base = np.e):
    #max = np.max(x,axis=1,keepdims=True) #returns max of each row and keeps same dims
    e_x = np.power(base,x) #subtracts each row with its max value
    e_x = e_x * (x > 0).astype(int)

    sum = np.sum(e_x,axis=1,keepdims=True) #returns sum of each row and keeps same dims
    f_x = e_x / sum
    return f_x

def write_zip(df,path,fname):
    if not os.path.isdir(path):
        os.makedirs(path)
    compression_options = {"method": "zip", "archive_name": f"{fname}.csv"}
    output_path = os.path.join(path,f"{fname}.zip")
    df.to_csv(output_path,compression=compression_options,index=False)

def normalize_rows(m,noise=.001):
    # Normalizes the cells in each row of a matrix to the total
    for i, row in enumerate(m):
        for j, col in enumerate(row):
            if m[i][j] ==0:
                m[i][j] += noise

        summing = sum(row)
        for j, col in enumerate(row):
            m[i][j] = m[i][j]/summing

def get_tau_distr(upper,lower,mu,sigma,nsamples):

    samples = stats.truncnorm.rvs(
        (lower-mu)/sigma,(upper-mu)/sigma,loc=mu,scale=sigma,size=nsamples)

    return samples