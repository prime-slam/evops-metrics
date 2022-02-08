from typing import Callable, Any

import numpy as np
from nptyping import NDArray

from src.utils.metrics import __group_indices_by_labels


def mean(
    pc_points: NDArray[(Any, 3), np.float64],
    pred_labels: NDArray[Any, np.int32],
    gt_labels: NDArray[Any, np.int32],
    metric: Callable[
        [NDArray[(Any, 3), np.float64], NDArray[Any, np.int32], NDArray[Any, np.int32]],
        np.float64,
    ],
) -> np.float64:
    """
    :param pc_points: source point cloud
    :param pred_labels: labels of points obtained as a result of segmentation
    :param gt_labels: reference labels of point cloud
    :param metric: metric function for which you want to get the mean value
    :return: list of mean value for each metric
    """
    assert (
        len(pc_points.shape) == 2 and pc_points.shape[1] == 3
    ), "Incorrect point cloud array size, expected (n, 3)"
    assert (
        len(pred_labels.shape) == 1
    ), "Incorrect predicted label array size, expected (n)"
    assert (
        len(gt_labels.shape) == 1
    ), "Incorrect ground truth label array size, expected (n)"
    assert (
        pc_points.shape[0] == pred_labels.size
    ), "Number of points does not match the array of predicted labels"
    assert (
        pc_points.shape[0] == gt_labels.size
    ), "Number of points does not match the array of ground truth labels"

    plane_predicted_dict = __group_indices_by_labels(pred_labels)
    plane_gt_dict = __group_indices_by_labels(gt_labels)
    unique_labels = np.unique(pred_labels)
    mean_array = np.empty((1, 0), np.float64)

    for label in unique_labels:
        if label in plane_gt_dict:
            mean_array = np.append(
                mean_array,
                metric(pc_points, plane_predicted_dict[label], plane_gt_dict[label]),
            )

    if mean_array.size == 0:
        return 0

    return mean_array.mean()
