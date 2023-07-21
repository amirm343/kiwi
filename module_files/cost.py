import numpy as np


def cross_entropy(target, prediction):
    l = np.multiply(target, prediction)
    return -np.log(l.max()).numpy()