from typing import Dict, Optional, Tuple


def quantitative_planes(
    assoc_dict: Dict[int, Optional[int]], planes_from_previous_frame: set[int]
) -> float:
    """
    Metric for calculating the ratio of number correctly associated planes to their total number
    :param assoc_dict: dictionary in
    {ID of plane from current frame: ID of plane from previous frame} format
    :param planes_from_previous_frame: set of planes from previous frame
    :return: metric result
    """
    right = 0
    for cur, prev in assoc_dict.items():
        if (cur == prev) or (prev is None and cur not in planes_from_previous_frame):
            right += 1
    return right / len(assoc_dict)


def quantitative_points(
    assoc_dict_with_length: Dict[Tuple[int, int], Optional[int]],
    planes_from_previous_frame: set,
) -> float:
    """
    Metric for calculating the ratio of number correctly associated points to their total number
    :param assoc_dict_with_length: dictionary in
    {(ID of plane from current frame, number of points in plane):
    ID of plane from previous frame} format
    :param planes_from_previous_frame: set of planes from previous frame
    :return: metric result
    """
    right = 0
    all_points = 0
    for (cur, cur_length), prev in assoc_dict_with_length.items():
        all_points += cur_length
        if (cur == prev) or (prev is None and cur not in planes_from_previous_frame):
            right += cur_length
    return right / all_points
