from typing import Any

import numpy as np
from nptyping import NDArray


def dice(
    pc_points: NDArray[(Any, 3), np.float64],
    pred_indices: NDArray[Any, np.int32],
    gt_indices: NDArray[Any, np.int32],
) -> np.float64:
    """
    :param pc_points: source point cloud
    :param pred_indices: indices of points that belong to one plane obtained as a result of segmentation
    :param gt_indices: indices of points belonging to the reference plane
    :return: iou metric value for plane
    """
    assert pred_indices.size + gt_indices.size != 0, "Array sizes must be positive"

    intersection = np.intersect1d(pred_indices, gt_indices)
    intersection_size = intersection.size
    gt_size = gt_indices.size
    predicted_size = pred_indices.size

    return 2 * intersection_size / (predicted_size + gt_size)
