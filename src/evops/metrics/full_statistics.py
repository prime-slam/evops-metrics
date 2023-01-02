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

from evops.benchmark.default import __recall, __precision, __fScore
from evops.benchmark.detailed import __detailed
from evops.benchmark.mean import __mean
from evops.benchmark.panoptic import __panoptic
from evops.utils.check_input import __tp_condition_assert, __pred_gt_assert


def full_statistics(
    pred_labels: NDArray[Any, np.int32],
    gt_labels: NDArray[Any, np.int32],
    metric: Callable[
        [NDArray[Any, np.int32], NDArray[Any, np.int32]],
        np.float64,
    ],
    tp_condition: str,
) -> Dict[str, float]:
    """
    :param pred_labels: array containing the labels of points obtained as a result of segmentation
    :param gt_labels: array containing the reference labels of point cloud
    :param metric: metric function for which you want to get the mean value
    Possible values from this library: metrics.iou and metrics.dice
    :param tp_condition: helper function to match planes from predicted to reference ones. Possible values: {'iou'}
    :return: dictionary with all supported plane detection metric names and their values
    """
    __pred_gt_assert(pred_labels, gt_labels)
    __tp_condition_assert(tp_condition)

    mean_result = __mean(pred_labels, gt_labels, metric, tp_condition)
    precision_result = __precision(pred_labels, gt_labels, tp_condition)
    recall_result = __recall(pred_labels, gt_labels, tp_condition)
    fScore_result = __fScore(pred_labels, gt_labels, tp_condition)
    panoptic_result = __panoptic(pred_labels, gt_labels, metric, tp_condition)
    usr, osr, missed, noise = __detailed(pred_labels, gt_labels, tp_condition).values()

    return {
        "panoptic": panoptic_result,
        "precision": precision_result,
        "recall": recall_result,
        "fScore": fScore_result,
        "usr": usr,
        "osr": osr,
        "noise": noise,
        "missed": missed,
        "mean": mean_result,
    }
