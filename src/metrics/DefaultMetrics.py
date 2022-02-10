from typing import Dict, Any

import numpy as np
from nptyping import NDArray

from src.utils.metrics import __group_indices_by_labels


def precision(
    pc_points: NDArray[(Any, 3), np.float64],
    pred_indices: NDArray[Any, np.int32],
    gt_indices: NDArray[Any, np.int32],
) -> np.float64:
    assert (
        len(pc_points.shape) == 2 and pc_points.shape[1] == 3
    ), "Incorrect point cloud array size, expected (n, 3)"
    assert (
        len(pred_indices.shape) == 1
    ), "Incorrect predicted label array size, expected (n)"
    assert (
        len(gt_indices.shape) == 1
    ), "Incorrect ground truth label array size, expected (n)"
    assert pred_indices.size != 0, "Predicted indices array size must not be zero"
    """
    :param pc_points: source point cloud
    :param pred_indices: indices of points that belong to one plane obtained as a result of segmentation
    :param gt_indices: indices of points belonging to the reference plane
    :return: precision metric value for plane
    """
    truePositive = np.intersect1d(pred_indices, gt_indices).size

    return truePositive / pred_indices.size


def accuracy(
    pc_points: NDArray[(Any, 3), np.float64],
    pred_indices: NDArray[Any, np.int32],
    gt_indices: NDArray[Any, np.int32],
) -> np.float64:
    assert pc_points.size != 0, "Source point cloud size must not be zero"
    assert (
        len(pc_points.shape) == 2 and pc_points.shape[1] == 3
    ), "Incorrect point cloud array size, expected (n, 3)"
    assert (
        len(pred_indices.shape) == 1
    ), "Incorrect predicted label array size, expected (n)"
    assert (
        len(gt_indices.shape) == 1
    ), "Incorrect ground truth label array size, expected (n)"
    """
    :param pc_points: source point cloud
    :param pred_indices: indices of points that belong to one plane obtained as a result of segmentation
    :param gt_indices: indices of points belonging to the reference plane
    :return: accuracy metric value for plane
    """
    truePositive = np.intersect1d(pred_indices, gt_indices).size
    trueNegative = pc_points.size - np.union1d(pred_indices, gt_indices).size

    return (truePositive + trueNegative) / pc_points.size


def recall(
    pc_points: NDArray[(Any, 3), np.float64],
    pred_indices: NDArray[Any, np.int32],
    gt_indices: NDArray[Any, np.int32],
) -> np.float64:
    assert (
        len(pc_points.shape) == 2 and pc_points.shape[1] == 3
    ), "Incorrect point cloud array size, expected (n, 3)"
    assert (
        len(pred_indices.shape) == 1
    ), "Incorrect predicted label array size, expected (n)"
    assert (
        len(gt_indices.shape) == 1
    ), "Incorrect ground truth label array size, expected (n)"
    assert gt_indices.size != 0, "Ground truth indices array size must not be zero"
    """
    :param pc_points: source point cloud
    :param pred_indices: indices of points that belong to one plane obtained as a result of segmentation
    :param gt_indices: indices of points belonging to the reference plane
    :return: recall metric value for plane
    """
    truePositive = np.intersect1d(pred_indices, gt_indices).size

    return truePositive / gt_indices.size


def fScore(
    pc_points: NDArray[(Any, 3), np.float64],
    pred_indices: NDArray[Any, np.int32],
    gt_indices: NDArray[Any, np.int32],
) -> np.float64:
    assert (
        len(pc_points.shape) == 2 and pc_points.shape[1] == 3
    ), "Incorrect point cloud array size, expected (n, 3)"
    assert (
        len(pred_indices.shape) == 1
    ), "Incorrect predicted label array size, expected (n)"
    assert (
        len(gt_indices.shape) == 1
    ), "Incorrect ground truth label array size, expected (n)"
    """
    :param pc_points: source point cloud
    :param pred_indices: indices of points that belong to one plane obtained as a result of segmentation
    :param gt_indices: indices of points belonging to the reference plane
    :return: f-score metric value for plane
    """
    precision_value = precision(pc_points, pred_indices, gt_indices)
    recall_value = recall(pc_points, pred_indices, gt_indices)

    if precision_value + recall_value == 0:
        return 0

    return 2 * precision_value * recall_value / (precision_value + recall_value)
