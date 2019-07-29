import math

def bool2int(x):
    """function for binary to decimal
    from: # https://stackoverflow.com/a/15505648/5916727"""
    y = 0
    for i,j in enumerate(x):
        y += j<<i
    return y


def sigmoid(x, x_0, k=8):
    # https://en.wikipedia.org/wiki/Logistic_function
    return 1 / (1 + math.exp(-k*(x-x_0)))
