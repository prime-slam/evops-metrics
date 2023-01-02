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

from typing import Any, Dict
from nptyping import NDArray

import evops.metrics.constants
from evops.utils.metrics_utils import __group_indices_by_labels, __statistics_functions

__USR_METRIC_NAME = "usr"
__OSR_METRIC_NAME = "osr"
__NOISE_METRIC_NAME = "noise"
__MISSED_METRIC_NAME = "missed"


def __usr(
    pred_labels: NDArray[Any, np.int32],
    gt_labels: NDArray[Any, np.int32],
    tp_condition: str,
) -> float:
    return __detailed(pred_labels, gt_labels, tp_condition)[__USR_METRIC_NAME]


def __osr(
    pred_labels: NDArray[Any, np.int32],
    gt_labels: NDArray[Any, np.int32],
    tp_condition: str,
) -> float:
    return __detailed(pred_labels, gt_labels, tp_condition)[__OSR_METRIC_NAME]


def __noise(
    pred_labels: NDArray[Any, np.int32],
    gt_labels: NDArray[Any, np.int32],
    tp_condition: str,
) -> float:
    return __detailed(pred_labels, gt_labels, tp_condition)[__NOISE_METRIC_NAME]


def __missed(
    pred_labels: NDArray[Any, np.int32],
    gt_labels: NDArray[Any, np.int32],
    tp_condition: str,
) -> float:
    return __detailed(pred_labels, gt_labels, tp_condition)[__MISSED_METRIC_NAME]


def __detailed(
    pred_labels: NDArray[Any, np.int32],
    gt_labels: NDArray[Any, np.int32],
    tp_condition: str,
) -> Dict[str, float]:
    predicted_label_to_indices = __group_indices_by_labels(pred_labels)
    gt_label_to_indices = __group_indices_by_labels(gt_labels)
    if evops.metrics.constants.UNSEGMENTED_LABEL in predicted_label_to_indices:
        del predicted_label_to_indices[evops.metrics.constants.UNSEGMENTED_LABEL]
    if evops.metrics.constants.UNSEGMENTED_LABEL in gt_label_to_indices:
        del gt_label_to_indices[evops.metrics.constants.UNSEGMENTED_LABEL]
    predicted_amount = len(predicted_label_to_indices)
    gt_amount = len(gt_label_to_indices)

    overlapped_gt_to_pred = {label: [] for label in gt_label_to_indices.keys()}
    part_overlapped_gt_to_pred = {label: [] for label in gt_label_to_indices.keys()}
    overlapped_pred_to_gt = {label: [] for label in predicted_label_to_indices.keys()}
    part_overlapped_pred_to_gt = {
        label: [] for label in predicted_label_to_indices.keys()
    }

    tp_condition_function = __statistics_functions[tp_condition]

    for predicted_label, predicted_indices in predicted_label_to_indices.items():
        for gt_label, gt_indices in gt_label_to_indices.items():
            are_well_overlapped = tp_condition_function(
                predicted_indices,
                gt_indices,
                evops.metrics.constants.IOU_THRESHOLD_FULL,
            )
            if are_well_overlapped:
                overlapped_pred_to_gt[predicted_label].append(gt_label)
                overlapped_gt_to_pred[gt_label].append(predicted_label)

            are_part_overlapped = tp_condition_function(
                predicted_indices,
                gt_indices,
                evops.metrics.constants.IOU_THRESHOLD_PART,
            )
            if are_part_overlapped:
                part_overlapped_pred_to_gt[predicted_label].append(gt_label)
                part_overlapped_gt_to_pred[gt_label].append(predicted_label)

    under_segmented_amount = 0
    noise_amount = 0
    for (
        predicted_label,
        part_overlapped_gt_planes,
    ) in part_overlapped_pred_to_gt.items():
        part_overlapped_gt_amount = len(part_overlapped_gt_planes)
        if part_overlapped_gt_amount > 1:
            under_segmented_amount += 1

        has_no_full_overlap = len(overlapped_pred_to_gt[predicted_label]) == 0
        has_single_only_part_overlap = (
            part_overlapped_gt_amount == 1 and has_no_full_overlap
        )
        if has_single_only_part_overlap:
            part_overlapped_gt_label = part_overlapped_gt_planes[0]
            is_not_target_of_osr = (
                len(part_overlapped_gt_to_pred[part_overlapped_gt_label]) == 1
            )
        else:
            is_not_target_of_osr = False
        if (
            part_overlapped_gt_amount == 0
            or is_not_target_of_osr
            and has_single_only_part_overlap
        ):
            noise_amount += 1

    missed_amount = 0
    over_segmented_amount = 0
    for gt_label, part_overlapped_pred_planes in part_overlapped_gt_to_pred.items():
        part_overlapped_pred_amount = len(part_overlapped_pred_planes)
        if part_overlapped_pred_amount > 1:
            over_segmented_amount += 1

        has_no_full_overlap = len(overlapped_gt_to_pred[gt_label]) == 0
        has_single_only_part_overlap = (
            part_overlapped_pred_amount == 1 and has_no_full_overlap
        )
        if has_single_only_part_overlap:
            part_overlapped_predicted_label = part_overlapped_pred_planes[0]
            is_not_target_of_usr = (
                len(part_overlapped_pred_to_gt[part_overlapped_predicted_label]) == 1
            )
        else:
            is_not_target_of_usr = False
        if (
            part_overlapped_pred_amount == 0
            or is_not_target_of_usr
            and has_single_only_part_overlap
        ):
            missed_amount += 1

    usr = under_segmented_amount / predicted_amount if predicted_amount != 0 else 0.0
    osr = over_segmented_amount / gt_amount if gt_amount != 0 else 0.0
    missed = missed_amount / gt_amount if gt_amount != 0 else 0.0
    noise = noise_amount / predicted_amount if predicted_amount != 0 else 0.0

    return {
        __USR_METRIC_NAME: usr,
        __OSR_METRIC_NAME: osr,
        __MISSED_METRIC_NAME: missed,
        __NOISE_METRIC_NAME: noise,
    }
