from typing import Optional


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
    assoc_dict_with_length: dict[tuple[int, int], Optional[int]],
    gt_assoc_dict: dict[int, Optional[int]],
) -> float:
    """
    Metric for calculating the ratio of number correctly associated points to their total number
    :param assoc_dict_with_length: dictionary in
    {(ID of plane from current frame, number of points in plane):
    ID of plane from previous frame} format
    :param gt_assoc_dict: groundtruth association dictionary in
    {ID of plane from current frame: ID of plane from previous frame} format
    :return: metric result
    """
    right = 0
    all_points = 0
    for (cur, cur_length), prev in assoc_dict_with_length.items():
        all_points += cur_length
        if gt_assoc_dict[cur] == prev:
            right += cur_length
    return right / all_points
