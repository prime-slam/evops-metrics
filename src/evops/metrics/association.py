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

from typing import Optional, Callable, Any, Dict
from nptyping import NDArray

from evops.benchmark.association import (
    __matched_points_ratio_with_matching,
    __matched_planes_ratio_with_matching,
    __matched_points_ratio,
    __matched_planes_ratio,
)
from evops.utils.check_input import __pred_gt_assert
from evops.utils.association_utils import match_labels_with_gt


def matched_planes_ratio(
    pred_assoc_dict: Dict[int, Optional[int]], gt_assoc_dict: Dict[int, Optional[int]]
) -> float:
    """
    Metric for calculating the ratio of number correctly associated planes to their total number
    REMEMBER: ground truth plane ids have to be synchronised with predicted ones
    If this isn't done use `quantitative_planes_with_matching` method
    :param pred_assoc_dict: dictionary in
    {ID of plane from current frame: ID of plane from previous frame} format with predicted associations
    :param gt_assoc_dict: ground truth association dictionary in
    {ID of plane from current frame: ID of plane from previous frame} format with ground truth associations
    :return: metric result
    """
    return __matched_planes_ratio(pred_assoc_dict, gt_assoc_dict)


def matched_points_ratio(
    pred_assoc_dict: Dict[int, Optional[int]],
    planes_sizes: Dict[int, int],
    gt_assoc_dict: Dict[int, Optional[int]],
) -> float:
    """
    Metric for calculating the ratio of number correctly associated points to their total number
    REMEMBER: ground truth plane ids have to be synchronised with predicted ones
    If this isn't done use `quantitative_points_with_matching` method
    :param pred_assoc_dict: dictionary in
    {ID of plane from current frame: ID of plane from previous frame} format with predicted associations
    :param gt_assoc_dict: ground truth association dictionary in
    {ID of plane from current frame: ID of plane from previous frame} format with ground truth associations
    :return: metric result
    """
    return __matched_points_ratio(pred_assoc_dict, planes_sizes, gt_assoc_dict)


def matched_planes_ratio_with_matching(
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
    """
    Metric for calculating the ratio of number correctly associated planes to their total number
    This method matches ground truth plane ids with predicted ones using `matcher` function
    REMEMBER: prediction quality can influence this metric
    because by default ground truth prediction is used as base for ground truth associations
    :param pred_assoc_dict: dictionary in
    {ID of plane from current frame: ID of plane from previous frame} format with predicted associations
    :param pred_labels_cur: predicted labels from current frame
    :param pred_labels_prev: predicted labels from previous frame
    :param gt_labels_cur: ground truth labels from current frame
    :param gt_labels_prev: ground truth labels from previous frame
    :param matcher: function that matches labels from current and previous frames using ground truth
    :return: metric result
    """
    __pred_gt_assert(pred_labels_cur, gt_labels_cur)
    __pred_gt_assert(pred_labels_prev, gt_labels_prev)

    return __matched_planes_ratio_with_matching(
        pred_assoc_dict,
        pred_labels_cur,
        pred_labels_prev,
        gt_labels_cur,
        gt_labels_prev,
        matcher,
    )


def matched_points_ratio_with_matching(
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
    """
    Metric for calculating the ratio of number correctly associated points to their total number
    This method matches ground truth plane ids with predicted ones using `matcher` function
    REMEMBER: prediction quality can influence this metric
    because by default ground truth prediction is used as base for ground truth associations
    :param pred_assoc_dict: dictionary in
    {(ID of plane from current frame, number of points in plane):
    ID of plane from previous frame} format with predicted associations
    :param pred_labels_cur: predicted labels from current frame
    :param pred_labels_prev: predicted labels from previous frame
    :param gt_labels_cur: ground truth labels from current frame
    :param gt_labels_prev: ground truth labels from previous frame
    :param matcher: function that matches labels from current and previous frames using ground truth
    :return: metric result
    """
    __pred_gt_assert(pred_labels_cur, gt_labels_cur)
    __pred_gt_assert(pred_labels_prev, gt_labels_prev)

    return __matched_points_ratio_with_matching(
        pred_assoc_dict,
        pred_labels_cur,
        pred_labels_prev,
        gt_labels_cur,
        gt_labels_prev,
        matcher,
    )
