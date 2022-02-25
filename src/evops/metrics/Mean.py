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
from typing import Any, Callable
from nptyping import NDArray
from evops.utils.metrics_utils import __group_indices_by_labels

import numpy as np
import evops.metrics.metrics


def __mean(
    pc_points: NDArray[(Any, 3), np.float64],
    pred_labels: NDArray[Any, np.int32],
    gt_labels: NDArray[Any, np.int32],
    metric: Callable[
        [NDArray[(Any, 3), np.float64], NDArray[Any, np.int32], NDArray[Any, np.int32]],
        np.float64,
    ],
) -> np.float64:
    plane_predicted_dict = __group_indices_by_labels(pred_labels)
    plane_gt_dict = __group_indices_by_labels(gt_labels)
    if evops.metrics.metrics.UNSEGMENTED_LABEL in plane_predicted_dict:
        del plane_predicted_dict[evops.metrics.metrics.UNSEGMENTED_LABEL]
    if evops.metrics.metrics.UNSEGMENTED_LABEL in plane_gt_dict:
        del plane_gt_dict[evops.metrics.metrics.UNSEGMENTED_LABEL]
    unique_labels = np.unique(pred_labels)
    mean_array = np.empty((1, 0), np.float64)

    for label in unique_labels:
        if label in plane_gt_dict:
            mean_array = np.append(
                mean_array,
                metric(pc_points, plane_predicted_dict[label], plane_gt_dict[label]),
            )

    if mean_array.size == 0:
        return 0

    return mean_array.mean()
