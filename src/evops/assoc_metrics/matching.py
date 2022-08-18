import numpy as np
from typing import Optional
from numpy.typing import NDArray


def __match_one_pair(labels, labeled_image):
    labeled_image = (
        labeled_image.reshape((labeled_image.shape[0] * labeled_image.shape[1], 3))
        / 255
    )
    annot_unique = np.unique(labels, axis=0)
    plane_color_dict = dict.fromkeys(annot_unique)
    del plane_color_dict[0]
    color_max_length_dict = dict()
    for annot_num in annot_unique:
        if annot_num == 0:
            continue
        indices = np.where(labels == annot_num)[0]
        colors, counts = np.unique(labeled_image[indices], axis=0, return_counts=True)
        if (colors[counts.argmax()] == 0).all():
            continue
        color = str(colors[counts.argmax()])
        if color in color_max_length_dict:
            if color_max_length_dict[color][0] > max(counts):
                continue
            plane_color_dict[color_max_length_dict[color][1]] = None
        color_max_length_dict[color] = max(counts), annot_num
        plane_color_dict[annot_num] = colors[counts.argmax()]
    return plane_color_dict


def match_labels_with_groundtruth(
    labels_cur: NDArray[np.uint8],
    labels_prev: NDArray[np.uint8],
    labeled_image_cur: NDArray[np.uint8],
    labeled_image_prev: NDArray[np.uint8],
) -> dict[int, Optional[int]]:
    """
    Matches labels from two frames using groundtruth
    :param labels_cur: labels from current frame
    :param labels_prev: labels from previous frame
    :param labeled_image_cur: current labeled groundtruth frame
    :param labeled_image_prev: previous labeled groundtruth frame
    :return: dictionary in
    {ID of plane from current frame: ID of plane from previous frame} format
    """
    cur_matched = __match_one_pair(labels_cur, labeled_image_cur)
    prev_matched = __match_one_pair(labels_prev, labeled_image_prev)
    result_dict = dict.fromkeys(cur_matched.keys())
    for cur_id, cur_color in cur_matched.items():
        if cur_color is None:
            continue
        for prev_id, prev_color in prev_matched.items():
            if (cur_color == prev_color).all():
                result_dict[cur_id] = prev_id
                break
    return result_dict
