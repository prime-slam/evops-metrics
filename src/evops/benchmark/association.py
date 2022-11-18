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
from typing import Dict, Optional, Any, Callable
from nptyping import NDArray

import numpy as np

from evops.utils.matching import match_labels_with_gt


def __quantitative_planes(
    pred_assoc_dict: Dict[int, Optional[int]],
    gt_assoc_dict: Dict[int, Optional[int]]
) -> float:
    right = 0
    for cur, prev in pred_assoc_dict.items():
        if gt_assoc_dict[cur] == prev:
            right += 1
    return right / len(pred_assoc_dict)


def __quantitative_points(
    pred_assoc_dict: Dict[int, Optional[int]],
    planes_sizes: Dict[int, int],
    gt_assoc_dict: Dict[int, Optional[int]],
) -> float:
    right = 0
    all_points = 0
    for cur, prev in pred_assoc_dict.items():
        cur_size = planes_sizes[cur]
        all_points += cur_size
        if gt_assoc_dict[cur] == prev:
            right += cur_size

    return right / all_points


def __quantitative_planes_with_matching(
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
    ] = match_labels_with_gt,
) -> float:
    gt_assoc_dict = matcher(pred_labels_cur, pred_labels_prev, gt_labels_cur, gt_labels_prev)
    return __quantitative_planes(pred_assoc_dict, gt_assoc_dict)


def __quantitative_points_with_matching(
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
    gt_assoc_dict = matcher(pred_labels_cur, pred_labels_prev, gt_labels_cur, gt_labels_prev)
    planes_sizes = dict()
    for cur in pred_assoc_dict.keys():
        planes_sizes[cur] = len(np.where(pred_labels_cur == cur)[0])

    return __quantitative_points(pred_assoc_dict, planes_sizes, gt_assoc_dict)
