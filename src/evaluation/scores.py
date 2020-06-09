import numpy as np


def dice2d(mask_gt, mask_seg):
    dice = 0
    for i in range(np.shape(mask_gt)[2]):
        dice += np.nan_to_num(
            2 * np.sum(np.logical_and(mask_gt[:, :, i], mask_seg[:, :, i])) /
            (np.sum(mask_gt[:, :, i]) + np.sum(mask_seg[:, :, i])))
    dice /= np.shape(mask_gt)[2]
    return dice


def dice(mask_gt, mask_seg):
    return 2 * np.sum(np.logical_and(
        mask_gt, mask_seg)) / (np.sum(mask_gt) + np.sum(mask_seg))
