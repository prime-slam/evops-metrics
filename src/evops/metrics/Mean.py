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
from evops.metrics import constants

import numpy as np


def __mean(
    pred_labels: NDArray[Any, np.int32],
    gt_labels: NDArray[Any, np.int32],
    metric: Callable[
        [NDArray[(Any, 3), np.float64], NDArray[Any, np.int32], NDArray[Any, np.int32]],
        np.float64,
    ],
) -> np.float64:
    plane_predicted_dict = __group_indices_by_labels(pred_labels)
    plane_gt_dict = __group_indices_by_labels(gt_labels)
    if constants.UNSEGMENTED_LABEL in plane_predicted_dict:
        del plane_predicted_dict[constants.UNSEGMENTED_LABEL]
    if constants.UNSEGMENTED_LABEL in plane_gt_dict:
        del plane_gt_dict[constants.UNSEGMENTED_LABEL]
    mean_array = np.zeros(len(plane_predicted_dict.keys()), np.float64)

    for label_index, label in enumerate(plane_predicted_dict.keys()):
        max_metric_value = 0
        for gt_label in plane_gt_dict.keys():
            metric_value = metric(plane_predicted_dict[label], plane_gt_dict[gt_label])
            max_metric_value = max(max_metric_value, metric_value)

        mean_array[label_index] = max_metric_value

    if mean_array.size == 0:
        return 0

    return mean_array.mean()
