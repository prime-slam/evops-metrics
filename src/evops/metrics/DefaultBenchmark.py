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
from typing import Callable, Any
from nptyping import NDArray

import numpy as np

import evops.metrics.constants


def __precision(
    pred_labels: NDArray[Any, np.int32],
    gt_labels: NDArray[Any, np.int32],
    statistic_func: Callable[
        [NDArray[Any, np.int32], NDArray[Any, np.int32]],
        np.int32,
    ],
) -> np.float64:
    pred_labels = np.delete(
        pred_labels,
        np.where(pred_labels == evops.metrics.constants.UNSEGMENTED_LABEL)[0],
    )
    assert (
        pred_labels.size != 0
    ), "Incorrect predicted label array values, most likely no labels other than UNSEGMENTED_LABEL"

    true_positive = statistic_func(pred_labels, gt_labels)
    pred_planes = np.unique(pred_labels)
    pred_planes = np.delete(
        pred_planes,
        np.where(pred_planes == evops.metrics.constants.UNSEGMENTED_LABEL)[0],
    )

    return true_positive / pred_planes.size


def __recall(
    pred_labels: NDArray[Any, np.int32],
    gt_labels: NDArray[Any, np.int32],
    statistic_func: Callable[
        [NDArray[Any, np.int32], NDArray[Any, np.int32]],
        np.int32,
    ],
) -> np.float64:
    gt_labels = np.delete(
        gt_labels, np.where(gt_labels == evops.metrics.constants.UNSEGMENTED_LABEL)[0]
    )
    assert (
        gt_labels.size != 0
    ), "Incorrect ground truth label array values, most likely no labels other than UNSEGMENTED_LABEL"

    true_positive = statistic_func(pred_labels, gt_labels)
    gt_planes = np.unique(gt_labels)
    gt_planes = np.delete(
        gt_planes,
        np.where(gt_planes == evops.metrics.constants.UNSEGMENTED_LABEL)[0],
    )

    return true_positive / gt_planes.size


def __fScore(
    pred_labels: NDArray[Any, np.int32],
    gt_labels: NDArray[Any, np.int32],
    statistic_func: Callable[
        [NDArray[Any, np.int32], NDArray[Any, np.int32]],
        np.int32,
    ],
) -> np.float64:
    precision = __precision(pred_labels, gt_labels, statistic_func)
    recall = __recall(pred_labels, gt_labels, statistic_func)

    return 2 * precision * recall / (precision + recall)
