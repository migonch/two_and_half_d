import numpy as np
from surface_distance.metrics import compute_surface_distances, compute_surface_dice_at_tolerance

from dpipe.batch_iter import apply_at


def multiclass_to_bool(metric, class_):
    def wrapper(multiclass_gt_and_spacing, multiclass_proba):
        multiclass_gt, spacing = multiclass_gt_and_spacing
        return metric((multiclass_gt == class_, spacing), np.argmax(multiclass_proba, 0) == class_)

    return wrapper


def binary_to_bool(metric):
    def wrapper(binary_gt_and_spacing, proba):
        return metric(apply_at(0, np.bool_)(binary_gt_and_spacing), proba >= .5)

    return wrapper


def drop_spacing(metric):
    def wrapper(gt_and_spacing, pred):
        gt, _ = gt_and_spacing
        return metric(gt, pred)

    return wrapper


def surface_dice(bool_gt_and_spacing, bool_pred, tolerance_mm=1):
    bool_gt, spacing = bool_gt_and_spacing
    return compute_surface_dice_at_tolerance(compute_surface_distances(bool_gt, bool_pred, spacing), tolerance_mm)
