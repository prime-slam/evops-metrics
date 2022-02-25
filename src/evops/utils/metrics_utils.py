from typing import Dict, Any

import numpy as np
from nptyping import NDArray


def __group_indices_by_labels(
    labels_array: NDArray[Any, np.int32],
) -> Dict[np.int32, NDArray[Any, np.int32]]:
    """
    :param labels_array: list of point cloud labels
    :return: dictionary with labels and an array of indices belonging to this label
    """
    unique_labels = np.unique(labels_array)
    dictionary = {}

    for label in unique_labels:
        label_indices = np.where(labels_array == label)[0]
        dictionary[label] = label_indices

    return dictionary


def __are_nearly_overlapped(
    plane_predicted: NDArray[Any, np.int32],
    plane_gt: NDArray[Any, np.int32],
    required_overlap: np.float64,
) -> (bool, bool):
    """
    Calculate if planes are overlapped enough (required_overlap %) to be used for PP-PR metric
    :param required_overlap: overlap threshold which will b checked to say that planes overlaps
    :param plane_predicted: predicted segmentation
    :param plane_gt: ground truth segmentation
    :return: true if planes are overlapping by required_overlap % or more, false otherwise
    """
    intersection = np.intersect1d(plane_predicted, plane_gt)

    return (
        intersection.size / plane_predicted.size >= required_overlap
        and intersection.size / plane_gt.size >= required_overlap,
        intersection.size > 0,
    )
