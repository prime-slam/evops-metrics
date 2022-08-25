import numpy as np
from typing import Optional, Callable, Any
from nptyping import NDArray
from evops.assoc_metrics.matching import match_labels_with_groundtruth


def quantitative_planes(
    assoc_dict: dict[int, Optional[int]], gt_assoc_dict: dict[int, Optional[int]]
) -> float:
    """
    Metric for calculating the ratio of number correctly associated planes to their total number
    :param assoc_dict: dictionary in
    {ID of plane from current frame: ID of plane from previous frame} format
    :param gt_assoc_dict: groundtruth association dictionary in
    {ID of plane from current frame: ID of plane from previous frame} format
    :return: metric result
    """
    right = 0
    for cur, prev in assoc_dict.items():
        if gt_assoc_dict[cur] == prev:
            right += 1
    return right / len(assoc_dict)


def quantitative_points(
    assoc_dict: dict[int, Optional[int]],
    planes_length: dict[int, int],
    gt_assoc_dict: dict[int, Optional[int]],
) -> float:
    """
    Metric for calculating the ratio of number correctly associated points to their total number
    :param assoc_dict: dictionary in
    {ID of plane from current frame: ID of plane from previous frame} format
    :param planes_length: dictionary in
    {ID of plane from current frame: number of points} format
    :param gt_assoc_dict: groundtruth association dictionary in
    {ID of plane from current frame: ID of plane from previous frame} format
    :return: metric result
    """
    right = 0
    all_points = 0
    for cur, prev in assoc_dict.items():
        cur_length = planes_length[cur]
        all_points += cur_length
        if gt_assoc_dict[cur] == prev:
            right += cur_length
    return right / all_points


def quantitative_planes_with_matching(
    assoc_dict: dict[int, Optional[int]],
    labels_cur: NDArray[(Any, Any), np.uint8],
    labels_prev: NDArray[(Any, Any), np.uint8],
    gt_labels_cur: NDArray[(Any, Any, 3), np.uint8],
    gt_labels_prev: NDArray[(Any, Any, 3), np.uint8],
    matcher: Callable[
        [
            NDArray[(Any, Any), np.uint8],
            NDArray[(Any, Any), np.uint8],
            NDArray[(Any, Any, 3), np.uint8],
            NDArray[(Any, Any, 3), np.uint8],
        ],
        dict[int, Optional[int]],
    ] = match_labels_with_groundtruth,
) -> float:
    """
    Metric for calculating the ratio of number correctly associated planes to their total number
    :param assoc_dict: dictionary in
    {ID of plane from current frame: ID of plane from previous frame} format
    :param labels_cur: labels from current frame
    :param labels_prev: labels from previous frame
    :param gt_labels_cur: current labeled groundtruth frame
    :param gt_labels_prev: previous labeled groundtruth frame
    :param matcher: func that matches labels from two frames using groundtruth
    :return: metric result
    """
    gt_assoc_dict = matcher(labels_cur, labels_prev, gt_labels_cur, gt_labels_prev)
    return quantitative_planes(assoc_dict, gt_assoc_dict)


def quantitative_points_with_matching(
    assoc_dict: dict[int, Optional[int]],
    labels_cur: NDArray[(Any, Any), np.uint8],
    labels_prev: NDArray[(Any, Any), np.uint8],
    gt_labels_cur: NDArray[(Any, Any, 3), np.uint8],
    gt_labels_prev: NDArray[(Any, Any, 3), np.uint8],
    matcher: Callable[
        [
            NDArray[(Any, Any), np.uint8],
            NDArray[(Any, Any), np.uint8],
            NDArray[(Any, Any, 3), np.uint8],
            NDArray[(Any, Any, 3), np.uint8],
        ],
        dict[int, Optional[int]],
    ] = match_labels_with_groundtruth,
) -> float:
    """
    Metric for calculating the ratio of number correctly associated points to their total number
    :param assoc_dict: dictionary in
    {(ID of plane from current frame, number of points in plane):
    ID of plane from previous frame} format
    :param labels_cur: labels from current frame
    :param labels_prev: labels from previous frame
    :param gt_labels_cur: current labeled groundtruth frame
    :param gt_labels_prev: previous labeled groundtruth frame
    :param matcher: func that matches labels from two frames using groundtruth
    :return: metric result
    """
    gt_assoc_dict = matcher(labels_cur, labels_prev, gt_labels_cur, gt_labels_prev)
    planes_length = dict()
    for cur, _ in assoc_dict.items():
        planes_length[cur] = len(np.where(labels_cur == cur)[0])
    return quantitative_points(assoc_dict, planes_length, gt_assoc_dict)
