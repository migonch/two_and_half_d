import numpy as np

from dpipe.batch_iter import unpack_args
from dpipe.im.slices import iterate_slices
from dpipe.itertools import lmap


def slicewisely(predict):
    def wrapper(*arrays):
        return np.stack(lmap(unpack_args(predict), iterate_slices(*arrays, axis=-1)), -1)

    return wrapper
