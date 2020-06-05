import random

import numpy as np

from dpipe.im import crop_to_box
from dpipe.im.box import get_centered_box
from dpipe.im.patch import sample_box_center_uniformly, get_random_patch


def get_random_slice(*arrays):
    slc = np.random.randint(arrays[0].shape[-1])
    return [a[..., slc] for a in arrays]


def get_random_slices(image, spacing, gt, n_slices):
    image, gt = get_random_patch(image, gt, patch_size=n_slices)
    return image, spacing, gt


def tumor_sampling(image, gt, patch_size, tumor_p=.5):
    def center_is_valid(center, box_size, shape):
        start, stop = get_centered_box(center, box_size)
        return np.all(start >= 0) and np.all(stop <= np.asarray(shape))

    center = random.choice(np.argwhere(gt > 0))
    if np.random.binomial(1, 1 - tumor_p) or not center_is_valid(center, patch_size, gt.shape):
        center = sample_box_center_uniformly(gt.shape, patch_size)

    box = get_centered_box(center, patch_size)
    return crop_to_box(image, box), crop_to_box(gt, box)
