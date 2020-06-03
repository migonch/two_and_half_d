import numpy as np


def to_binary(metric, class_):
    def wrapper(gt, prediction):
        return metric(gt == class_, np.argmax(prediction, 0) == class_)

    return wrapper
