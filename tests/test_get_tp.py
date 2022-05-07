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
import pytest

import evops.metrics.constants
from evops.utils.MetricsUtils import __get_tp


def test_get_tp_using_iou():
    evops.metrics.constants.IOU_THRESHOLD = 0.75
    evops.metrics.constants.UNSEGMENTED_LABEL = 0

    pred_labels = np.array([1, 1, 3, 3])
    gt_labels = np.array([2, 2, 0, 3])

    assert 1 == __get_tp(pred_labels, gt_labels, "iou")


def test_get_tp_null_using_iou():
    evops.metrics.constants.IOU_THRESHOLD = 0.75
    evops.metrics.constants.UNSEGMENTED_LABEL = 0

    pred_labels = np.array([1, 2, 3, 4])
    gt_labels = np.array([1, 1, 1, 1])

    assert 0 == __get_tp(pred_labels, gt_labels, "iou")


def test_get_tp_full_using_iou():
    evops.metrics.constants.IOU_THRESHOLD = 0.75
    evops.metrics.constants.UNSEGMENTED_LABEL = 0

    pred_labels = np.array([1, 2, 3, 4])
    gt_labels = np.array([1, 2, 3, 4])

    assert 4 == __get_tp(pred_labels, gt_labels, "iou")
