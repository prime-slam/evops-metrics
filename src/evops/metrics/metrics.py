from typing import Callable, Any
from nptyping import NDArray

from evops.metrics.DefaultMetrics import __precision, __accuracy, __recall, __fScore
from evops.metrics.DiceBenchmark import __dice
from evops.metrics.IoUBenchmark import __iou
from evops.metrics.MultiValueBenchmark import __multi_value_benchmark
from evops.metrics.Mean import __mean

import numpy as np


UNSEGMENTED_LABEL = 0


def iou(
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
    assert (
        len(pc_points.shape) == 2 and pc_points.shape[1] == 3
    ), "Incorrect point cloud array size, expected (n, 3)"
    assert (
        len(pred_indices.shape) == 1
    ), "Incorrect predicted label array size, expected (n)"
    assert (
        len(gt_indices.shape) == 1
    ), "Incorrect ground truth label array size, expected (n)"
    assert pc_points.size != 0, "Point cloud size must be positive"
    assert pred_indices.size + gt_indices.size != 0, "Array sizes must be positive"

    return __iou(pc_points, pred_indices, gt_indices)


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
    assert (
        len(pc_points.shape) == 2 and pc_points.shape[1] == 3
    ), "Incorrect point cloud array size, expected (n, 3)"
    assert (
        len(pred_indices.shape) == 1
    ), "Incorrect predicted label array size, expected (n)"
    assert (
        len(gt_indices.shape) == 1
    ), "Incorrect ground truth label array size, expected (n)"
    assert pc_points.size != 0, "Point cloud size must be positive"
    assert pred_indices.size + gt_indices.size != 0, "Array sizes must be positive"

    return __dice(pc_points, pred_indices, gt_indices)


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
    assert (
        len(pc_points.shape) == 2 and pc_points.shape[1] == 3
    ), "Incorrect point cloud array size, expected (n, 3)"
    assert (
        len(pred_indices.shape) == 1
    ), "Incorrect predicted label array size, expected (n)"
    assert (
        len(gt_indices.shape) == 1
    ), "Incorrect ground truth label array size, expected (n)"
    assert pc_points.size != 0, "Point cloud size must be positive"
    assert pred_indices.size != 0, "Predicted indices array size must not be zero"

    return __precision(pc_points, pred_indices, gt_indices)


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
    assert (
        len(pc_points.shape) == 2 and pc_points.shape[1] == 3
    ), "Incorrect point cloud array size, expected (n, 3)"
    assert (
        len(pred_indices.shape) == 1
    ), "Incorrect predicted label array size, expected (n)"
    assert (
        len(gt_indices.shape) == 1
    ), "Incorrect ground truth label array size, expected (n)"
    assert pc_points.size != 0, "Point cloud size must be positive"

    return __accuracy(pc_points, pred_indices, gt_indices)


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
    assert (
        len(pc_points.shape) == 2 and pc_points.shape[1] == 3
    ), "Incorrect point cloud array size, expected (n, 3)"
    assert (
        len(pred_indices.shape) == 1
    ), "Incorrect predicted label array size, expected (n)"
    assert (
        len(gt_indices.shape) == 1
    ), "Incorrect ground truth label array size, expected (n)"
    assert pc_points.size != 0, "Point cloud size must be positive"
    assert gt_indices.size != 0, "Ground truth indices array size must not be zero"

    return __recall(pc_points, pred_indices, gt_indices)


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
    assert (
        len(pc_points.shape) == 2 and pc_points.shape[1] == 3
    ), "Incorrect point cloud array size, expected (n, 3)"
    assert (
        len(pred_indices.shape) == 1
    ), "Incorrect predicted label array size, expected (n)"
    assert (
        len(gt_indices.shape) == 1
    ), "Incorrect ground truth label array size, expected (n)"
    assert pc_points.size != 0, "Point cloud size must be positive"
    assert gt_indices.size != 0, "Ground truth indices array size must not be zero"

    return __fScore(pc_points, pred_indices, gt_indices)


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
    assert pc_points.size != 0, "Point cloud size must be positive"
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

    return __mean(pc_points, pred_labels, gt_labels, metric)


def multi_value(
    pc_points: NDArray[(Any, 3), np.float64],
    pred_labels: NDArray[Any, np.int32],
    gt_labels: NDArray[Any, np.int32],
    overlap_threshold: np.float64 = 0.8,
) -> (np.float64, np.float64, np.float64, np.float64, np.float64, np.float64):
    """
    :param pc_points: source point cloud
    :param pred_labels: labels of points obtained as a result of segmentation
    :param gt_labels: reference labels of point cloud
    :param overlap_threshold: minimum value at which the planes are considered intersected
    :return: precision, recall, under_segmented, over_segmented, missed, noise
    """
    assert (
        len(pc_points.shape) == 2 and pc_points.shape[1] == 3
    ), "Incorrect point cloud array size, expected (n, 3)"
    assert pc_points.size != 0, "Point cloud size must be positive"
    assert (
        len(pred_labels.shape) == 1
    ), "Incorrect predicted label array size, expected (n)"
    assert (
        len(gt_labels.shape) == 1
    ), "Incorrect ground truth label array size, expected (n)"

    return __multi_value_benchmark(pc_points, pred_labels, gt_labels, overlap_threshold)
