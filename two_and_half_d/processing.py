import numpy as np


def multiclass_to_binary(array, labels):
    return np.stack([array == label for label in labels])

