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
import pytest

import evops.metrics.constants
from evops.utils.iou_overlap import __is_overlapped_iou

from fixtures import clean_env


def test_full_iou_overlap(clean_env):
    pred_indices = np.array([1, 2, 3, 4])
    gt_indices = np.array([1, 2, 3, 4])

    assert __is_overlapped_iou(pred_indices, gt_indices)


def test_null_iou_overlap(clean_env):
    pred_indices = np.array([1, 2, 3, 4])
    gt_indices = np.array([5, 6, 7, 8])

    assert not __is_overlapped_iou(pred_indices, gt_indices)


def test_iou_overlap(clean_env):
    evops.metrics.constants.IOU_THRESHOLD_FULL = 0.25

    pred_indices = np.array([1, 2, 3, 4])
    gt_indices = np.array([1, 2, 5, 6])

    assert __is_overlapped_iou(pred_indices, gt_indices)
