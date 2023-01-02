# Copyright (c) 2022, Pavel Mokeev, Dmitrii Iarosh, Anastasiia Kornilova, Ivan Moskalenko
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

from evops.metrics.constants import UNSEGMENTED_LABEL
from typing import Optional, Any, Dict
from nptyping import NDArray


def __match_planes_by_max_overlap(
    pred_labels: NDArray[Any, np.int32], gt_labels: NDArray[Any, np.int32]
) -> Dict[int, Optional[int]]:
    """
    Matches predicted and ground truth labels by maximum overlap of planes
    :param pred_labels: predicted labels for frame
    :param gt_labels: ground truth labels for frame
    :return: dictionary in
    {ID of plane from predicted labels : ID of plane from ground truth labels} format
    """
    pred_labels_unique = np.unique(pred_labels, axis=0)
    matches_with_overlap_size = dict()

    # Getting plane ID: (color, number of points) pairs
    for pred_label in pred_labels_unique:
        if pred_label == UNSEGMENTED_LABEL:
            continue

        (label_indices,) = np.where(pred_labels == pred_label)
        overlapped_gt_labels, counts = np.unique(
            gt_labels[label_indices], return_counts=True
        )
        matched_gt_label = overlapped_gt_labels[counts.argmax()]
        if matched_gt_label == UNSEGMENTED_LABEL:
            continue

        matches_with_overlap_size[pred_label] = matched_gt_label, max(counts)

    # Sorting matches by overlap size
    sorted_matches = sorted(
        matches_with_overlap_size.items(), key=lambda x: x[1][1], reverse=True
    )
    used_labels = set()
    result = dict.fromkeys(pred_labels_unique)
    del result[UNSEGMENTED_LABEL]

    # Filling the resulting dict with control of the colors used
    for (pred_label, (gt_label, _)) in sorted_matches:
        if gt_label in used_labels:
            continue
        result[pred_label] = gt_label
        used_labels.add(gt_label)

    return result


def match_labels_with_gt(
    pred_labels_cur: NDArray[Any, np.int32],
    pred_labels_prev: NDArray[Any, np.int32],
    gt_labels_cur: NDArray[Any, np.int32],
    gt_labels_prev: NDArray[Any, np.int32],
) -> Dict[int, Optional[int]]:
    """
    Matches labels from two frames using associated ground truth
    :param pred_labels_cur: predicted labels from current frame
    :param pred_labels_prev: predicted labels from previous frame
    :param gt_labels_cur: ground truth labels for current frame
    :param gt_labels_prev: ground truth labels for previous frame
    :return: dictionary in
    {ID of predicted plane from current frame: ID of predicted plane from previous frame} format
    """
    cur_pred_to_gt = __match_planes_by_max_overlap(pred_labels_cur, gt_labels_cur)
    prev_pred_to_gt = __match_planes_by_max_overlap(pred_labels_prev, gt_labels_prev)
    result_dict = dict.fromkeys(cur_pred_to_gt.keys())
    for cur_pred_label, cur_gt_label in cur_pred_to_gt.items():
        if cur_gt_label is None:
            continue
        for prev_pred_label, prev_gt_label in prev_pred_to_gt.items():
            if cur_gt_label == prev_gt_label:
                result_dict[cur_pred_label] = prev_pred_label
                break

    return result_dict
