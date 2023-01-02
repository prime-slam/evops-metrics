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

from typing import Dict, Any
from nptyping import NDArray

import evops.metrics.constants
from evops.utils.iou_overlap import __is_overlapped_iou

__statistics_functions = {"iou": __is_overlapped_iou}


def __group_indices_by_labels(
    labels_array: NDArray[Any, np.int32],
) -> Dict[np.int32, NDArray[Any, np.int32]]:
    """
    :param labels_array: list of point cloud labels
    :return: dictionary with labels and an array of indices belonging to this label
    """
    unique_labels = np.unique(labels_array)
    dictionary = {}

    for label in unique_labels:
        (label_indices,) = np.where(labels_array == label)
        dictionary[label] = label_indices

    return dictionary


def __calc_tp(
    pred_labels: NDArray[Any, np.int32],
    gt_labels: NDArray[Any, np.int32],
    tp_condition: str,
) -> np.int32:
    """
    :param pred_labels: labels of points corresponding to segmented planes
    :param gt_labels: labels of points corresponding to ground truth planes
    :param tp_condition: helper function to calculate statistics
    :return: true positive received using pred_labels and gt_labels
    """
    true_positive = 0

    unique_gt_labels = __filter_unsegmented(np.unique(gt_labels))
    unique_pred_labels = __filter_unsegmented(np.unique(pred_labels))

    pred_used = set()
    tp_condition_function = __statistics_functions[tp_condition]

    for gt_label in unique_gt_labels:
        (gt_indices,) = np.where(gt_labels == gt_label)
        for pred_label in unique_pred_labels:
            if pred_label in pred_used:
                continue

            (pred_indices,) = np.where(pred_labels == pred_label)
            is_overlap = tp_condition_function(pred_indices, gt_indices)

            if is_overlap:
                true_positive += 1
                pred_used.add(pred_label)
                break

    return true_positive


def __filter_unsegmented(
    label_array: NDArray[Any, np.int32],
) -> NDArray[Any, np.int32]:
    """
    :param label_array: labels of points corresponding to segmented planes
    :return: labels array where all unsegmented points (with label equal to evops.metrics.constants.UNSEGMENTED_LABEL)
     are deleted
    """
    (unsegmented_indices,) = np.where(
        label_array == evops.metrics.constants.UNSEGMENTED_LABEL
    )
    label_array = np.delete(
        label_array,
        unsegmented_indices,
    )

    return label_array
