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

from typing import Dict, Optional, Any, Callable
from nptyping import NDArray


def __matched_planes_ratio(
    pred_assoc_dict: Dict[int, Optional[int]], gt_assoc_dict: Dict[int, Optional[int]]
) -> float:
    right = 0
    for label_from_cur_frame, label_from_prev_frame in pred_assoc_dict.items():
        if gt_assoc_dict[label_from_cur_frame] == label_from_prev_frame:
            right += 1
    return right / len(pred_assoc_dict)


def __matched_points_ratio(
    pred_assoc_dict: Dict[int, Optional[int]],
    planes_sizes: Dict[int, int],
    gt_assoc_dict: Dict[int, Optional[int]],
) -> float:
    right = 0
    all_points = 0
    for label_from_cur_frame, label_from_prev_frame in pred_assoc_dict.items():
        cur_size = planes_sizes[label_from_cur_frame]
        all_points += cur_size
        if gt_assoc_dict[label_from_cur_frame] == label_from_prev_frame:
            right += cur_size

    return right / all_points


def __matched_planes_ratio_with_matching(
    pred_assoc_dict: Dict[int, Optional[int]],
    pred_labels_cur: NDArray[Any, np.int32],
    pred_labels_prev: NDArray[Any, np.int32],
    gt_labels_cur: NDArray[Any, np.int32],
    gt_labels_prev: NDArray[Any, np.int32],
    matcher: Callable[
        [
            NDArray[Any, np.int32],
            NDArray[Any, np.int32],
            NDArray[Any, np.int32],
            NDArray[Any, np.int32],
        ],
        Dict[int, Optional[int]],
    ],
) -> float:
    gt_assoc_dict = matcher(
        pred_labels_cur, pred_labels_prev, gt_labels_cur, gt_labels_prev
    )
    return __matched_planes_ratio(pred_assoc_dict, gt_assoc_dict)


def __matched_points_ratio_with_matching(
    pred_assoc_dict: Dict[int, Optional[int]],
    pred_labels_cur: NDArray[Any, np.int32],
    pred_labels_prev: NDArray[Any, np.int32],
    gt_labels_cur: NDArray[Any, np.int32],
    gt_labels_prev: NDArray[Any, np.int32],
    matcher: Callable[
        [
            NDArray[Any, np.int32],
            NDArray[Any, np.int32],
            NDArray[Any, np.int32],
            NDArray[Any, np.int32],
        ],
        Dict[int, Optional[int]],
    ],
) -> float:
    gt_assoc_dict = matcher(
        pred_labels_cur, pred_labels_prev, gt_labels_cur, gt_labels_prev
    )
    planes_sizes = dict()
    for label_from_cur_frame in pred_assoc_dict.keys():
        planes_sizes[label_from_cur_frame] = len(
            np.where(pred_labels_cur == label_from_cur_frame)[0]
        )

    return __matched_points_ratio(pred_assoc_dict, planes_sizes, gt_assoc_dict)
