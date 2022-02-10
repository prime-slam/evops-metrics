from typing import Any, Dict

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
    assert pred_indices.size != 0, "predicted indices array size must not be zero"
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
    falsePositive = np.union1d(pred_indices, gt_indices).size - gt_indices.size
    falseNegative = np.union1d(pred_indices, gt_indices).size - pred_indices.size

    return (truePositive + trueNegative) / (
        truePositive + trueNegative + falsePositive + falseNegative
    )


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
    assert gt_indices.size != 0, "ground truth indices array size must not be zero"
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


def get_all_multi_value_metrics(
    pc_points: NDArray[(Any, 3), np.float64],
    pred_labels: NDArray[Any, np.int32],
    gt_labels: NDArray[Any, np.int32],
) -> Dict[np.int64, NDArray[4, np.float64]]:
    assert (
        len(pc_points.shape) == 2 and pc_points.shape[1] == 3
    ), "Incorrect point cloud array size, expected (n, 3)"
    assert (
        len(pred_labels.shape) == 1
    ), "Incorrect predicted label array size, expected (n)"
    assert (
        len(gt_labels.shape) == 1
    ), "Incorrect ground truth label array size, expected (n)"
    """
    :param pc_points: source point cloud
    :param pred_labels: labels of points obtained as a result of segmentation
    :param gt_labels: reference labels of point cloud
    :return: dictionary with keys --- labels, values --- array with metric values
    """
    plane_predicted_dict = __group_indices_by_labels(pred_labels)
    plane_gt_dict = __group_indices_by_labels(gt_labels)
    metrics_dictionary = {}

    for label in plane_predicted_dict:
        if label in plane_gt_dict:
            precision_value = precision(
                pc_points, plane_predicted_dict[label], plane_gt_dict[label]
            )
            accuracy_value = accuracy(
                pc_points, plane_predicted_dict[label], plane_gt_dict[label]
            )
            recall_value = recall(
                pc_points, plane_predicted_dict[label], plane_gt_dict[label]
            )
            fScore_value = fScore(
                pc_points, plane_predicted_dict[label], plane_gt_dict[label]
            )
            metrics_dictionary[label] = np.array(
                [precision_value, accuracy_value, recall_value, fScore_value]
            )

    return metrics_dictionary
