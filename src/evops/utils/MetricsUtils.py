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
from typing import Dict, Any
from nptyping import NDArray

import numpy as np

import evops.metrics.constants
from evops.utils.IoUOverlap import __is_overlapped_iou

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
        label_indices = np.where(labels_array == label)[0]
        dictionary[label] = label_indices

    return dictionary


def __are_nearly_overlapped(
    predicted_indices: NDArray[Any, np.int32],
    gt_indices: NDArray[Any, np.int32],
    full_overlap_threshold: float,
    part_overlap_threshold: float,
    tp_condition: str,
) -> (bool, bool):
    """
    Calculate if planes are overlapped enough with IoU to be used for PP-PR metric
    :param full_overlap_threshold: overlap threshold which will be checked to say that planes overlaps fully
    :param part_overlap_threshold: overlap threshold which will be checked to say that planes overlaps partly
    :param predicted_indices: indices of points belonging to the given predicted label
    :param gt_indices: indices of points belonging to the given gt label
    :param tp_condition: helper function to calculate statistics
    :return: Two booleans
    1) true if planes are overlapping fully by full_overlap_threshold threshold or more, false otherwise
    2) true if planes are overlapping partly by part_overlap_threshold threshold or more, false otherwise
    """
    tp_condition_function = __statistics_functions[tp_condition]

    return (
        tp_condition_function(predicted_indices, gt_indices, full_overlap_threshold),
        tp_condition_function(predicted_indices, gt_indices, part_overlap_threshold),
    )


def __get_tp(
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
        gt_indices = np.where(gt_labels == gt_label)[0]
        for pred_label in unique_pred_labels:
            if pred_label in pred_used:
                continue

            pred_indices = np.where(pred_labels == pred_label)[0]
            is_overlap = tp_condition_function(pred_indices, gt_indices)

            if is_overlap:
                true_positive += 1
                pred_used.add(pred_label)
                break

    return true_positive


def __filter_unsegmented(
    label_array: NDArray[Any, np.int32],
) -> NDArray[Any, np.int32]:
    label_array = np.delete(
        label_array,
        np.where(label_array == evops.metrics.constants.UNSEGMENTED_LABEL)[0],
    )
    assert (
        label_array.size != 0
    ), "Incorrect label array values, most likely no labels other than UNSEGMENTED_LABEL"

    return label_array
