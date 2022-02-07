import numpy as np


def group_indices_by_labels(labels_array: np.ndarray) -> dict:
    unique_labels = np.unique(labels_array)
    dictionary = {}

    for label in unique_labels:
        label_indices = np.where(labels_array == label)[0]
        dictionary[label] = label_indices

    return dictionary
