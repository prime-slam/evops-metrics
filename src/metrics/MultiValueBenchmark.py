from typing import Any, Dict

import numpy as np
from nptyping import NDArray

from src.utils.metrics import __group_indices_by_labels


def precision(
    pc_points: NDArray[(Any, 3), np.float64],
    pred_indices: NDArray[Any, np.int32],
    gt_indices: NDArray[Any, np.int32],
) -> np.float64:
    """
    :param pc_points: source point cloud
    :param pred_indices: indices of points that belong to one plane obtained as a result of segmentation
    :param gt_indices: indices of points belonging to the reference plane
    :return: precision metric value for plane
    """
    TP = np.intersect1d(pred_indices, gt_indices).size
    FP = np.union1d(pred_indices, gt_indices).size - gt_indices.size

    return TP / FP


def accuracy(
    pc_points: NDArray[(Any, 3), np.float64],
    pred_indices: NDArray[Any, np.int32],
    gt_indices: NDArray[Any, np.int32],
) -> np.float64:
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
    """
    :param pc_points: source point cloud
    :param pred_indices: indices of points that belong to one plane obtained as a result of segmentation
    :param gt_indices: indices of points belonging to the reference plane
    :return: recall metric value for plane
    """
    truePositive = np.intersect1d(pred_indices, gt_indices).size
    falseNegative = np.union1d(pred_indices, gt_indices).size - pred_indices.size

    return truePositive / (truePositive + falseNegative)


def fScore(
    pc_points: NDArray[(Any, 3), np.float64],
    pred_indices: NDArray[Any, np.int32],
    gt_indices: NDArray[Any, np.int32],
) -> np.float64:
    """
    :param pc_points: source point cloud
    :param pred_indices: indices of points that belong to one plane obtained as a result of segmentation
    :param gt_indices: indices of points belonging to the reference plane
    :return: f-score metric value for plane
    """
    precision_value = precision(pc_points, pred_indices, gt_indices)
    recall_value = recall(pc_points, pred_indices, gt_indices)

    return 2 * precision_value * recall_value / (precision_value + recall_value)


def get_all_multi_value_metrics(
    pc_points: NDArray[(Any, 3), np.float64],
    pred_labels: NDArray[Any, np.int32],
    gt_labels: NDArray[Any, np.int32],
) -> Dict[np.int64, NDArray[4, np.float64]]:
    plane_predicted_dict = __group_indices_by_labels(pred_labels)
    plane_gt_dict = __group_indices_by_labels(gt_labels)
    metrics_dictionary = {}

    for label, plane in plane_predicted_dict:
        if label in plane_gt_dict:
            precision_value = precision(pc_points, pred_labels, gt_labels)
            accuracy_value = accuracy(pc_points, pred_labels, gt_labels)
            recall_value = recall(pc_points, pred_labels, gt_labels)
            fScore_value = fScore(pc_points, pred_labels, gt_labels)
            metrics_dictionary[label] = np.array(
                [precision_value, accuracy_value, recall_value, fScore_value]
            )

    return metrics_dictionary
