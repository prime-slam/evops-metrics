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
from evops.metrics.IoUBenchmark import __iou


def __iou_overlap(
    pred_labels: NDArray[Any, np.int32],
    gt_labels: NDArray[Any, np.int32],
) -> np.int32:
    """
    :param pred_labels: labels of points corresponding to segmented planes
    :param gt_labels: indices of points corresponding to ground truth planes
    :return: (true positive, false positive, false negative) received using pred_indices and gt_indices
    """
    true_positive = 0
    unique_gt_labels = np.unique(gt_labels)
    unique_gt_labels = np.delete(
        unique_gt_labels,
        np.where(unique_gt_labels == evops.metrics.constants.UNSEGMENTED_LABEL)[0],
    )
    unique_pred_labels = np.unique(pred_labels)
    unique_pred_labels = np.delete(
        unique_pred_labels,
        np.where(unique_pred_labels == evops.metrics.constants.UNSEGMENTED_LABEL)[0],
    )
    pred_used = set()

    for gt_label in unique_gt_labels:
        is_already_true_positive = False
        gt_indices = np.where(gt_labels == gt_label)[0]
        for pred_label in unique_pred_labels:
            pred_indices = np.where(pred_labels == pred_label)[0]

            IoU_value = __iou(pred_indices, gt_indices)

            if (
                IoU_value >= evops.metrics.constants.IOU_THRESHOLD
                and pred_label not in pred_used
                and not is_already_true_positive
            ):
                true_positive += 1
                is_already_true_positive = True
                pred_used.add(pred_label)

    return true_positive
