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
import numpy as np

from typing import Any
from nptyping import NDArray

import evops.metrics.constants
from evops.benchmark.iou import __iou


def __is_overlapped_iou(
    pred_indices: NDArray[Any, np.int32],
    gt_indices: NDArray[Any, np.int32],
    threshold: float = None,
) -> bool:
    """
    :param pred_indices: indices of points belonging to the given predicted label
    :param gt_indices: indices of points belonging to the given predicted label
    :param threshold: value at which planes will be selected as overlapped enough
    :return: true if IoU >= evops.metrics.constants.IOU_THRESHOLD_FULL
    """
    if threshold is None:
        threshold = evops.metrics.constants.IOU_THRESHOLD_FULL

    return __iou(pred_indices, gt_indices) >= threshold
