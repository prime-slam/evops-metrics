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

from typing import Any
from nptyping import NDArray

from evops.utils.metrics_utils import __filter_unsegmented, __calc_tp


def __precision(
    pred_labels: NDArray[Any, np.int32],
    gt_labels: NDArray[Any, np.int32],
    tp_condition: str,
) -> np.float64:
    true_positive = __calc_tp(pred_labels, gt_labels, tp_condition)
    pred_labels = __filter_unsegmented(pred_labels)

    unique_pred_labels_size = np.unique(pred_labels).size

    return (
        true_positive / unique_pred_labels_size if unique_pred_labels_size != 0 else 0.0
    )


def __recall(
    pred_labels: NDArray[Any, np.int32],
    gt_labels: NDArray[Any, np.int32],
    tp_condition: str,
) -> np.float64:
    true_positive = __calc_tp(pred_labels, gt_labels, tp_condition)
    gt_labels = __filter_unsegmented(gt_labels)

    unique_gt_labels_size = np.unique(gt_labels).size

    return true_positive / unique_gt_labels_size if unique_gt_labels_size != 0 else 0.0


def __fScore(
    pred_labels: NDArray[Any, np.int32],
    gt_labels: NDArray[Any, np.int32],
    tp_condition: str,
) -> np.float64:
    precision = __precision(pred_labels, gt_labels, tp_condition)
    recall = __recall(pred_labels, gt_labels, tp_condition)

    numerator = 2 * precision * recall
    denominator = precision + recall

    return numerator / denominator if denominator != 0 else 0.0
