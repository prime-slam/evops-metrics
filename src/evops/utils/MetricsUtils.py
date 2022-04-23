# Copyright (c) 2022, Pavel Mokeev, Dmitrii Iarosh, Anastasiia Kornilova
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
from typing import Dict, Any
from nptyping import NDArray

import numpy as np

from evops.utils.IoUOverlap import __iou_overlap

__statistics_functions = {"iou": __iou_overlap}


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
