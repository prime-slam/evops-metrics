# Copyright (c) 2022, Skolkovo Institute of Science and Technology (Skoltech)
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

from evops.utils.metrics_utils import __statistics_functions


def __tp_condition_assert(
    tp_condition: str,
):
    assert tp_condition in __statistics_functions, "Incorrect name of tp condition"


def __pred_gt_assert(
    pred_labels: NDArray[Any, np.int32],
    gt_labels: NDArray[Any, np.int32],
    require_equal_sizes=True,
):
    assert (
        len(pred_labels.shape) == 1
    ), "Incorrect predicted label array size, expected (n)"
    assert (
        len(gt_labels.shape) == 1
    ), "Incorrect ground truth label array size, expected (n)"
    assert (
        pred_labels.size + gt_labels.size != 0
    ), "Prediction and ground truth array sizes must be positive"
    if require_equal_sizes:
        assert (
            pred_labels.size == gt_labels.size
        ), "Prediction and ground truth array sizes must be equals"
