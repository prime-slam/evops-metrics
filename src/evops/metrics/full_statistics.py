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
from typing import Callable, Any, Dict
from nptyping import NDArray

import numpy as np

from evops.metrics.DefaultBenchmark import __precision, __recall, __fScore
from evops.metrics.DetailedBenchmark import __detailed
from evops.metrics.MeanBenchmark import __mean
from evops.metrics.PanopticBenchmark import __panoptic
from evops.utils.CheckInput import __default_benchmark_asserts


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
    :param pred_labels: labels of points obtained as a result of segmentation
    :param gt_labels: reference labels of point cloud
    :param metric: metric function for which you want to get the mean value
    :param tp_condition: helper function to calculate statistics: {'iou'}
    :return: dictionary with all supported metric names and their values
    """
    __default_benchmark_asserts(pred_labels, gt_labels, tp_condition)

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
