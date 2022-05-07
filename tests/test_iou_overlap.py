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
from evops.utils.IoUOverlap import __iou_overlap


def test_full_iou_overlap():
    evops.metrics.constants.IOU_THRESHOLD = 0.75

    pred_indices = np.array([1, 2, 3, 4])
    gt_indices = np.array([1, 2, 3, 4])

    assert __iou_overlap(pred_indices, gt_indices)


def test_null_iou_overlap():
    evops.metrics.constants.IOU_THRESHOLD = 0.75

    pred_indices = np.array([1, 2, 3, 4])
    gt_indices = np.array([5, 6, 7, 8])

    assert not __iou_overlap(pred_indices, gt_indices)


def test_iou_overlap():
    evops.metrics.constants.IOU_THRESHOLD = 0.25

    pred_indices = np.array([1, 2, 3, 4])
    gt_indices = np.array([1, 2, 5, 6])

    assert __iou_overlap(pred_indices, gt_indices)
