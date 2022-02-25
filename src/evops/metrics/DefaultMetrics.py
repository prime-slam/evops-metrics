from typing import Dict, Any

import numpy as np
from nptyping import NDArray

from src.utils.metrics_utils import __group_indices_by_labels


def __precision(
    pc_points: NDArray[(Any, 3), np.float64],
    pred_indices: NDArray[Any, np.int32],
    gt_indices: NDArray[Any, np.int32],
) -> np.float64:
    truePositive = np.intersect1d(pred_indices, gt_indices).size

    return truePositive / pred_indices.size


def __accuracy(
    pc_points: NDArray[(Any, 3), np.float64],
    pred_indices: NDArray[Any, np.int32],
    gt_indices: NDArray[Any, np.int32],
) -> np.float64:
    truePositive = np.intersect1d(pred_indices, gt_indices).size
    trueNegative = pc_points.size - np.union1d(pred_indices, gt_indices).size

    return (truePositive + trueNegative) / pc_points.size


def __recall(
    pc_points: NDArray[(Any, 3), np.float64],
    pred_indices: NDArray[Any, np.int32],
    gt_indices: NDArray[Any, np.int32],
) -> np.float64:
    truePositive = np.intersect1d(pred_indices, gt_indices).size

    return truePositive / gt_indices.size


def __fScore(
    pc_points: NDArray[(Any, 3), np.float64],
    pred_indices: NDArray[Any, np.int32],
    gt_indices: NDArray[Any, np.int32],
) -> np.float64:
    precision_value = __precision(pc_points, pred_indices, gt_indices)
    recall_value = __recall(pc_points, pred_indices, gt_indices)

    if precision_value + recall_value == 0:
        return 0

    return 2 * precision_value * recall_value / (precision_value + recall_value)
