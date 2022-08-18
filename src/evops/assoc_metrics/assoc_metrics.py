import numpy as np
from typing import Optional, Callable
from numpy.typing import NDArray
from matching import match_labels_with_groundtruth


def quantitative_planes(
    assoc_dict: dict[int, Optional[int]],
    labels_cur: NDArray[np.uint8],
    labels_prev: NDArray[np.uint8],
    labeled_image_cur: NDArray[np.uint8],
    labeled_image_prev: NDArray[np.uint8],
    matcher: Callable[
        [NDArray[np.uint8], NDArray[np.uint8], NDArray[np.uint8], NDArray[np.uint8]],
        dict[int, Optional[int]],
    ] = match_labels_with_groundtruth,
) -> float:
    """
    Metric for calculating the ratio of number correctly associated planes to their total number
    :param assoc_dict: dictionary in
    {ID of plane from current frame: ID of plane from previous frame} format
    :param labels_cur: labels from current frame
    :param labels_prev: labels from previous frame
    :param labeled_image_cur: current labeled groundtruth frame
    :param labeled_image_prev: previous labeled groundtruth frame
    :param matcher: func that matches labels from two frames using groundtruth
    :return: metric result
    """
    gt_assoc_dict = matcher(
        labels_cur, labels_prev, labeled_image_cur, labeled_image_prev
    )
    right = 0
    for cur, prev in assoc_dict.items():
        if gt_assoc_dict[cur] == prev:
            right += 1
    return right / len(assoc_dict)


def quantitative_points(
    assoc_dict: dict[int, Optional[int]],
    labels_cur: NDArray[np.uint8],
    labels_prev: NDArray[np.uint8],
    labeled_image_cur: NDArray[np.uint8],
    labeled_image_prev: NDArray[np.uint8],
    matcher: Callable[
        [NDArray[np.uint8], NDArray[np.uint8], NDArray[np.uint8], NDArray[np.uint8]],
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
    :param labeled_image_cur: current labeled groundtruth frame
    :param labeled_image_prev: previous labeled groundtruth frame
    :param matcher: func that matches labels from two frames using groundtruth
    :return: metric result
    """
    gt_assoc_dict = matcher(
        labels_cur, labels_prev, labeled_image_cur, labeled_image_prev
    )
    right = 0
    all_points = 0
    for cur, prev in assoc_dict.items():
        cur_length = len(np.where(labels_cur == cur)[0])
        all_points += cur_length
        if gt_assoc_dict[cur] == prev:
            right += cur_length
    return right / all_points
