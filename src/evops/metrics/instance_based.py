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
import numpy as np

from typing import Callable, Any, Dict
from nptyping import NDArray

from evops.benchmark.default import __precision, __recall, __fScore
from evops.benchmark.detailed import __usr, __noise, __missed, __osr
from evops.benchmark.panoptic import __panoptic
from evops.utils.check_input import __tp_condition_assert, __pred_gt_assert


def precision(
    pred_labels: NDArray[Any, np.int32],
    gt_labels: NDArray[Any, np.int32],
    tp_condition: str,
) -> np.float64:
    """
    :param pred_labels: array containing the labels of points obtained as a result of segmentation
    :param gt_labels: array containing the reference labels of point cloud
    :param tp_condition: helper function to match planes from predicted to reference ones. Possible values: {'iou'}
    :return: calculated precision
    """
    __pred_gt_assert(pred_labels, gt_labels)
    __tp_condition_assert(tp_condition)

    return __precision(pred_labels, gt_labels, tp_condition)


def recall(
    pred_labels: NDArray[Any, np.int32],
    gt_labels: NDArray[Any, np.int32],
    tp_condition: str,
) -> np.float64:
    """
    :param pred_labels: array containing the labels of points obtained as a result of segmentation
    :param gt_labels: array containing the reference labels of point cloud
    :param tp_condition: helper function to match planes from predicted to reference ones. Possible values: {'iou'}
    :return: calculated recall
    """
    __pred_gt_assert(pred_labels, gt_labels)
    __tp_condition_assert(tp_condition)

    return __recall(pred_labels, gt_labels, tp_condition)


def fScore(
    pred_labels: NDArray[Any, np.int32],
    gt_labels: NDArray[Any, np.int32],
    tp_condition: str,
) -> np.float64:
    """
    :param pred_labels: array containing the labels of points obtained as a result of segmentation
    :param gt_labels: array containing the reference labels of point cloud
    :param tp_condition: helper function to match planes from predicted to reference ones. Possible values: {'iou'}
    :return: calculated f-score
    """
    __pred_gt_assert(pred_labels, gt_labels)
    __tp_condition_assert(tp_condition)

    return __fScore(pred_labels, gt_labels, tp_condition)


def panoptic(
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
    :param metric: metric function for which mean value will be calculated.
    It is used for pixel perfect part of this metric.
    Possible values from this library: metrics.iou and metrics.dice
    :param tp_condition: helper function to match planes from predicted to reference ones. Possible values: {'iou'}
    :return: calculated panoptic metric value
    """
    __pred_gt_assert(pred_labels, gt_labels)
    __tp_condition_assert(tp_condition)

    return __panoptic(pred_labels, gt_labels, metric, tp_condition)


def usr(
    pred_labels: NDArray[Any, np.int32],
    gt_labels: NDArray[Any, np.int32],
    tp_condition: str,
) -> float:
    """
    :param pred_labels: array containing the labels of points obtained as a result of segmentation
    :param gt_labels: array containing the reference labels of point cloud
    :param tp_condition: helper function to match planes from predicted to reference ones. Possible values: {'iou'}
    :return: calculated undersegmentation ratio
    """
    __pred_gt_assert(pred_labels, gt_labels)
    __tp_condition_assert(tp_condition)

    return __usr(pred_labels, gt_labels, tp_condition)


def osr(
    pred_labels: NDArray[Any, np.int32],
    gt_labels: NDArray[Any, np.int32],
    tp_condition: str,
) -> float:
    """
    :param pred_labels: array containing the labels of points obtained as a result of segmentation
    :param gt_labels: array containing the reference labels of point cloud
    :param tp_condition: helper function to match planes from predicted to reference ones. Possible values: {'iou'}
    :return: calculated oversegmentation ratio
    """
    __pred_gt_assert(pred_labels, gt_labels)
    __tp_condition_assert(tp_condition)

    return __osr(pred_labels, gt_labels, tp_condition)


def noise(
    pred_labels: NDArray[Any, np.int32],
    gt_labels: NDArray[Any, np.int32],
    tp_condition: str,
) -> float:
    """
    :param pred_labels: array containing the labels of points obtained as a result of segmentation
    :param gt_labels: array containing the reference labels of point cloud
    :param tp_condition: helper function to match planes from predicted to reference ones. Possible values: {'iou'}
    :return: calculated noise planes ratio --- ratio of planes which are detected but don't exist in gt
    """
    __pred_gt_assert(pred_labels, gt_labels)
    __tp_condition_assert(tp_condition)

    return __noise(pred_labels, gt_labels, tp_condition)


def missed(
    pred_labels: NDArray[Any, np.int32],
    gt_labels: NDArray[Any, np.int32],
    tp_condition: str,
) -> float:
    """
    :param pred_labels: array containing the labels of points obtained as a result of segmentation
    :param gt_labels: array containing the reference labels of point cloud
    :param tp_condition: helper function to match planes from predicted to reference ones. Possible values: {'iou'}
    :return: calculated missed planes ratio --- ratio of planes which exist in gt but aren't detected
    """
    __pred_gt_assert(pred_labels, gt_labels)
    __tp_condition_assert(tp_condition)

    return __missed(pred_labels, gt_labels, tp_condition)
