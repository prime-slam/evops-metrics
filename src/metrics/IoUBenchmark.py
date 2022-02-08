import numpy as np


def iou(
    pc_points: np.ndarray, pred_indices: np.ndarray, gt_indices: np.ndarray
) -> np.float64:
    intersection = np.intersect1d(pred_indices, gt_indices)
    union = np.union1d(pred_indices, gt_indices)
    return intersection.size / union.size
