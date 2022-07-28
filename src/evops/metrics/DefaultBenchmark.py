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
from typing import Any
from nptyping import NDArray

import numpy as np

import evops.metrics.constants
from evops.utils.MetricsUtils import __get_tp, __filter_unsegmented


def __precision(
    pred_labels: NDArray[Any, np.int32],
    gt_labels: NDArray[Any, np.int32],
    tp_condition: str,
) -> np.float64:
    true_positive = __get_tp(pred_labels, gt_labels, tp_condition)
    pred_labels = __filter_unsegmented(pred_labels)

    return true_positive / np.unique(pred_labels).size


def __recall(
    pred_labels: NDArray[Any, np.int32],
    gt_labels: NDArray[Any, np.int32],
    tp_condition: str,
) -> np.float64:
    true_positive = __get_tp(pred_labels, gt_labels, tp_condition)
    gt_labels = __filter_unsegmented(gt_labels)

    return true_positive / np.unique(gt_labels).size


def __fScore(
    pred_labels: NDArray[Any, np.int32],
    gt_labels: NDArray[Any, np.int32],
    tp_condition: str,
) -> np.float64:
    precision = __precision(pred_labels, gt_labels, tp_condition)
    recall = __recall(pred_labels, gt_labels, tp_condition)

    # Prevent division by zero
    if precision + recall == 0:
        return 0.0

    return 2 * precision * recall / (precision + recall)
