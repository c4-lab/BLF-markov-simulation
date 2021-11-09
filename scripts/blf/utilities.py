import math
import numpy as np
from scipy.stats import pearsonr

def bool2int(x):
    """function for binary to decimal
    from: # https://stackoverflow.com/a/15505648/5916727"""
    y = 0
    for i,j in enumerate(x):
        y += j<<i
    return y


def int2bool(x,width = None):
    """function for decimal to binary"""
    if not width:
        return [int(x) for x in reversed(list(np.binary_repr(x)))]
    else:
        return [int(x) for x in reversed(list(np.binary_repr(x,width)))]

def sigmoid(x, x_0, k=8):
    # https://en.wikipedia.org/wiki/Logistic_function
    return 1 / (1 + math.exp(-k*(x-x_0)))

def hamming(a,b):
    return(bin(a^b).count("1"))


def measure_landscape_complexity(attractor_matrix):
    # Jones & Forrest 1995
    diag = attractor_matrix.diagonal()
    best_val = min(diag)
    global_attractors = [i for i, v in enumerate(diag) if v == best_val]

    f = []
    d = []
    for i,j in enumerate(diag):
        if j == best_val:
            continue
        f.append(j)
        d.append(min([hamming(i,g) for g in global_attractors]))

    return f,d,pearsonr(f,d)



def normalize_rows(m,noise=.001):
    # Normalizes the cells in each row of a matrix to the total
    for i, row in enumerate(m):
        for j, col in enumerate(row):
            if m[i][j] ==0:
                m[i][j] += noise

        summing = sum(row)
        for j, col in enumerate(row):
            m[i][j] = m[i][j]/summing

