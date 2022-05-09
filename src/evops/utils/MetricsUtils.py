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
from evops.utils.IoUOverlap import is_overlapped_iou

__statistics_functions = {"iou": is_overlapped_iou}


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
    plane_predicted: NDArray[Any, np.int32],
    plane_gt: NDArray[Any, np.int32],
    required_overlap: np.float64,
) -> (bool, bool):
    """
    Calculate if planes are overlapped enough (required_overlap %) to be used for PP-PR metric
    :param required_overlap: overlap threshold which will b checked to say that planes overlaps
    :param plane_predicted: predicted segmentation
    :param plane_gt: ground truth segmentation
    :return: true if planes are overlapping by required_overlap % or more, false otherwise
    """
    intersection = np.intersect1d(plane_predicted, plane_gt)

    return (
        intersection.size / plane_predicted.size >= required_overlap
        and intersection.size / plane_gt.size >= required_overlap,
        intersection.size > 0,
    )


def __filter_unsegmented_unique(
    label_array: NDArray[Any, np.int32],
) -> NDArray[Any, np.int32]:
    unique_label_array = np.unique(label_array)
    unique_label_array = np.delete(
        unique_label_array,
        np.where(unique_label_array == evops.metrics.constants.UNSEGMENTED_LABEL)[0],
    )
    assert (
        unique_label_array.size != 0
    ), "Incorrect labels unique count, most likely no labels other than UNSEGMENTED_LABEL"

    return unique_label_array


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

    unique_gt_labels = __filter_unsegmented_unique(gt_labels)
    unique_pred_labels = __filter_unsegmented_unique(pred_labels)

    pred_used = set()
    tp_condition_function = __statistics_functions[tp_condition]

    for gt_label in unique_gt_labels:
        gt_indices = np.where(gt_labels == gt_label)[0]
        for pred_label in unique_pred_labels:
            if pred_label in pred_used:
                continue

            pred_indices = np.where(pred_labels == pred_label)[0]
            is_overlap = tp_condition_function(pred_indices, gt_indices)

            if is_overlap and pred_label not in pred_used:
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
