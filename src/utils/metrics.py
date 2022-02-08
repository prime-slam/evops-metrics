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
