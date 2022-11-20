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
from typing import Any, Callable
from nptyping import NDArray

import numpy as np

from evops.benchmark.dice import __dice
from evops.benchmark.iou import __iou
from evops.benchmark.mean import __mean
from evops.utils.check_input import (
    __iou_dice_mean_bechmark_asserts,
    __default_benchmark_asserts,
)


def iou(
    pred_indices: NDArray[Any, np.int32],
    gt_indices: NDArray[Any, np.int32],
) -> np.float64:
    """
    :param pred_indices: array containing indices of the plane corresponding points as a result of segmentation
    :param gt_indices: array containing indices of the plane corresponding points as ground truth segmentation
    :return: IoU value for planes
    """
    __iou_dice_mean_bechmark_asserts(pred_indices, gt_indices)

    return __iou(pred_indices, gt_indices)


def dice(
    pred_indices: NDArray[Any, np.int32],
    gt_indices: NDArray[Any, np.int32],
) -> np.float64:
    """
    :param pred_indices: array containing indices of the plane corresponding points as a result of segmentation
    :param gt_indices: array containing indices of the plane corresponding points as ground truth segmentation
    :return: dice coefficient for planes
    """
    __iou_dice_mean_bechmark_asserts(pred_indices, gt_indices)

    return __dice(pred_indices, gt_indices)


def mean(
    pred_labels: NDArray[Any, np.int32],
    gt_labels: NDArray[Any, np.int32],
    metric: Callable[
        [NDArray[Any, np.int32], NDArray[Any, np.int32]],
        np.float64,
    ],
    tp_condition: str,
) -> float:
    """
    :param pred_labels: array containing the labels of points obtained as a result of segmentation
    :param gt_labels: array containing the reference labels of point cloud
    :param metric: metric function for which you want to get the mean value
    Possible values from this library: metrics.iou and metrics.dice
    :param tp_condition: helper function to match planes from predicted to reference ones. Possible values: {'iou'}
    :return: mean value of the selected metric calculated only for the matched planes
    """
    __default_benchmark_asserts(pred_labels, gt_labels, tp_condition)

    return __mean(pred_labels, gt_labels, metric, tp_condition)
