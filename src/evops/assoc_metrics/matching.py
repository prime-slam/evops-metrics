import numpy as np
from evops.metrics.constants import UNSEGMENTED_LABEL
from typing import Optional, Any
from nptyping import NDArray


def __match_one_pair(labels, gt_labeled_image):
    gt_labeled_image = (
        gt_labeled_image.reshape(
            (gt_labeled_image.shape[0] * gt_labeled_image.shape[1], 3)
        )
        / 255
    )
    annot_unique = np.unique(labels, axis=0)
    plane_color_length = dict()
    # Getting plane ID: (color, number of points) pairs
    for annot in annot_unique:
        if annot == UNSEGMENTED_LABEL:
            continue
        indices = np.where(labels == annot)[0]
        colors, counts = np.unique(
            gt_labeled_image[indices], axis=0, return_counts=True
        )
        matched_color = colors[counts.argmax()]
        if np.all(matched_color == 0):
            continue
        plane_color_length[annot] = matched_color, max(counts)
    # Sorting pairs by number of points
    sorted_length = sorted(
        plane_color_length.items(), key=lambda x: x[1][1], reverse=True
    )
    used_colors = set()
    result = dict.fromkeys(annot_unique)
    del result[UNSEGMENTED_LABEL]
    # Filling the resulting dict with control of the colors used
    for (plane_id, (color, _)) in sorted_length:
        color_str = str(color)
        if color_str in used_colors:
            continue
        result[plane_id] = color
        used_colors.add(color_str)
    return result


def match_labels_with_groundtruth(
    labels_cur: NDArray[(Any, Any), np.uint8],
    labels_prev: NDArray[(Any, Any), np.uint8],
    gt_labels_cur: NDArray[(Any, Any, 3), np.uint8],
    gt_labels_prev: NDArray[(Any, Any, 3), np.uint8],
) -> dict[int, Optional[int]]:
    """
    Matches labels from two frames using groundtruth
    :param labels_cur: labels from current frame
    :param labels_prev: labels from previous frame
    :param gt_labels_cur: current labeled groundtruth frame
    :param gt_labels_prev: previous labeled groundtruth frame
    :return: dictionary in
    {ID of plane from current frame: ID of plane from previous frame} format
    """
    cur_matched = __match_one_pair(labels_cur, gt_labels_cur)
    prev_matched = __match_one_pair(labels_prev, gt_labels_prev)
    result_dict = dict.fromkeys(cur_matched.keys())
    for cur_id, cur_color in cur_matched.items():
        if cur_color is None:
            continue
        for prev_id, prev_color in prev_matched.items():
            if (cur_color == prev_color).all():
                result_dict[cur_id] = prev_id
                break
    return result_dict
