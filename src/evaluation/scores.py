import numpy as np


def dice(mask_gt, mask_seg):
    return 2 * np.sum(np.logical_and(
        mask_gt, mask_seg)) / (np.sum(mask_gt) + np.sum(mask_seg))
