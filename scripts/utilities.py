import math

def bool2int(x):
    """function for binary to decimal
    from: # https://stackoverflow.com/a/15505648/5916727"""
    y = 0
    for i,j in enumerate(x):
        y += j<<i
    return y


def sigmoid(x):
    # https://stackoverflow.com/questions/3985619/how-to-calculate-a-logistic-sigmoid-function-in-python
    return 1 / (1 + math.exp(-x))
