# Copyright (c) 2022, Pavel Mokeev, Dmitrii Iarosh, Anastasiia Kornilova
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
from typing import Callable, Any
from nptyping import NDArray

from evops.metrics.DefaultMetrics import __precision, __accuracy, __recall, __fScore
from evops.metrics.DiceBenchmark import __dice
from evops.metrics.IoUBenchmark import __iou
from evops.metrics.MultiValueBenchmark import __multi_value_benchmark
from evops.metrics.Mean import __mean

import numpy as np


def iou(
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
        len(pred_indices.shape) == 1
    ), "Incorrect predicted label array size, expected (n)"
    assert (
        len(gt_indices.shape) == 1
    ), "Incorrect ground truth label array size, expected (n)"
    assert pred_indices.size + gt_indices.size != 0, "Array sizes must be positive"

    return __iou(pred_indices, gt_indices)


def dice(
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
        len(pred_indices.shape) == 1
    ), "Incorrect predicted label array size, expected (n)"
    assert (
        len(gt_indices.shape) == 1
    ), "Incorrect ground truth label array size, expected (n)"
    assert pred_indices.size + gt_indices.size != 0, "Array sizes must be positive"

    return __dice(pred_indices, gt_indices)


def precision(
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
        len(pred_indices.shape) == 1
    ), "Incorrect predicted label array size, expected (n)"
    assert (
        len(gt_indices.shape) == 1
    ), "Incorrect ground truth label array size, expected (n)"
    assert pred_indices.size != 0, "Predicted indices array size must not be zero"

    return __precision(pred_indices, gt_indices)


def accuracy(
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
        len(pred_indices.shape) == 1
    ), "Incorrect predicted label array size, expected (n)"
    assert (
        len(gt_indices.shape) == 1
    ), "Incorrect ground truth label array size, expected (n)"

    return __accuracy(pred_indices, gt_indices)


def recall(
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
        len(pred_indices.shape) == 1
    ), "Incorrect predicted label array size, expected (n)"
    assert (
        len(gt_indices.shape) == 1
    ), "Incorrect ground truth label array size, expected (n)"
    assert gt_indices.size != 0, "Ground truth indices array size must not be zero"

    return __recall(pred_indices, gt_indices)


def fScore(
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
        len(pred_indices.shape) == 1
    ), "Incorrect predicted label array size, expected (n)"
    assert (
        len(gt_indices.shape) == 1
    ), "Incorrect ground truth label array size, expected (n)"
    assert gt_indices.size != 0, "Ground truth indices array size must not be zero"

    return __fScore(pred_indices, gt_indices)


def mean(
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
        len(pred_labels.shape) == 1
    ), "Incorrect predicted label array size, expected (n)"
    assert (
        len(gt_labels.shape) == 1
    ), "Incorrect ground truth label array size, expected (n)"

    return __mean(pred_labels, gt_labels, metric)


def multi_value(
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
        len(pred_labels.shape) == 1
    ), "Incorrect predicted label array size, expected (n)"
    assert (
        len(gt_labels.shape) == 1
    ), "Incorrect ground truth label array size, expected (n)"

    return __multi_value_benchmark(pred_labels, gt_labels, overlap_threshold)
