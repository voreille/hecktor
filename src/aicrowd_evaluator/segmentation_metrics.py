import numpy as np
import SimpleITK as sitk

from surface_distance import compute_surface_distances, compute_robust_hausdorff


def compute_segmentation_scores(y_true, y_pred, spacing):
    if np.sum(y_pred) == 0:
        return {
            "dice_score": 0,
            "hausdorff_distance_95": 0,
            "recall": 0,
            "precision": 0,
        }
    else:
        fp = np.sum(np.logical_and(np.logical_not(y_true), y_pred))
        tp = np.sum(np.logical_and(y_true, y_pred))
        fn = np.sum(np.logical_and(np.logical_not(y_pred), y_true))
        # tn = np.sum(
        #     np.logical_and(np.logical_not(y_true), np.logical_not(y_pred)))
        recall = tp / (tp + fn)
        precision = tp / (tp + fp)
        return {
            "dice_score":
            dice(y_true, y_pred),
            "hausdorff_distance_95":
            robust_hausdorff(y_true != 0, y_pred != 0, spacing, percent=95),
            "recall":
            recall,
            "precision":
            precision
        }


def robust_hausdorff(image0, image1, spacing, percent=95.0):
    surface_distances = compute_surface_distances(image0 != 0, image1 != 0,
                                                  spacing)
    return compute_robust_hausdorff(surface_distances, percent)


def dice(y_true, y_pred):
    return 2 * np.sum(np.logical_and(
        y_true, y_pred)) / (np.sum(y_true) + np.sum(y_pred))


def get_np_volume_from_sitk(sitk_image):
    trans = (2, 1, 0)
    pixel_spacing = sitk_image.GetSpacing()
    image_position_patient = sitk_image.GetOrigin()
    np_image = sitk.GetArrayFromImage(sitk_image)
    np_image = np.transpose(np_image, trans)
    return np_image, pixel_spacing, image_position_patient
