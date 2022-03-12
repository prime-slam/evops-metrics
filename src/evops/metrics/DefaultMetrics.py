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
from typing import Any
from nptyping import NDArray
from sklearn.metrics import precision_score, accuracy_score, recall_score, f1_score

import numpy as np


def __precision(
    pred_labels: NDArray[Any, np.int32],
    gt_labels: NDArray[Any, np.int32],
) -> np.float64:
    return precision_score(gt_labels, pred_labels, average="micro")


def __accuracy(
    pred_labels: NDArray[Any, np.int32],
    gt_labels: NDArray[Any, np.int32],
) -> np.float64:
    return accuracy_score(gt_labels, pred_labels)


def __recall(
    pred_labels: NDArray[Any, np.int32],
    gt_labels: NDArray[Any, np.int32],
) -> np.float64:
    return recall_score(gt_labels, pred_labels, average="micro")


def __fScore(
    pred_labels: NDArray[Any, np.int32],
    gt_labels: NDArray[Any, np.int32],
) -> np.float64:
    return f1_score(gt_labels, pred_labels, average="micro")
