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

from evops.metrics.IoUBenchmark import __iou
from evops.metrics.constants import IOU_THRESHOLD


def __iou_overlap(
    pred_labels: NDArray[Any, np.int32],
    gt_labels: NDArray[Any, np.int32],
) -> (np.int32, np.int32, np.int32):
    """
    :param pred_labels: labels of points corresponding to segmented planes
    :param gt_labels: indices of points corresponding to ground truth planes
    :return: (true positive, false positive, false negative) received using pred_indices and gt_indices
    """
    true_positive = 0
    false_positive = 0
    false_negative = 0
    unique_gt_labels = np.unique(gt_labels)
    unique_gt_labels = np.delete(unique_gt_labels, np.where(unique_gt_labels == 0)[0])
    unique_pred_labels = np.unique(pred_labels)
    unique_pred_labels = np.delete(
        unique_pred_labels, np.where(unique_pred_labels == 0)[0]
    )

    for gt_label in unique_gt_labels:
        is_already_true_positive = False
        gt_indices = np.where(gt_labels == gt_label)[0]
        for pred_label in unique_pred_labels:
            pred_indices = np.where(pred_labels == pred_label)[0]

            IoU_value = __iou(pred_indices, gt_indices)

            if IoU_value == 0:
                false_negative += 1
            elif IoU_value >= IOU_THRESHOLD:
                if not is_already_true_positive:
                    true_positive += 1
                    is_already_true_positive = True
                else:
                    false_positive += 1
            else:
                false_positive += 1

    return true_positive, false_positive, false_negative
