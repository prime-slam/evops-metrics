import numpy as np


def get_dictionary_indices_of_current_label(indices_array: np.ndarray) -> dict:
    dictionary = {}

    for index, value in enumerate(indices_array):
        if value in dictionary:
            dictionary[value] = np.append(dictionary[value], index)
        else:
            dictionary[value] = np.array(index)

    return dictionary


def planes_intersection_indices(
    plane_left: np.ndarray, plane_right: np.ndarray
) -> np.array:
    return np.intersect1d(plane_left, plane_right)


def planes_union_indices(
    plane_left: np.ndarray,
    plane_right: np.ndarray,
) -> np.array:
    return np.union1d(plane_left, plane_right)
