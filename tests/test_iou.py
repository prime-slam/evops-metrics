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

from evops.metrics import iou
from evops.utils.metrics_utils import __group_indices_by_labels

from fixtures import clean_env


def test_null_iou_result(clean_env):
    pred_indices = np.array([1, 2, 3, 4])
    gt_indices = np.array([5, 6, 7, 8])

    assert 0 == pytest.approx(iou(pred_indices, gt_indices))


def test_full_iou_result(clean_env):
    pred_indices = np.array([1, 2, 3, 4])
    gt_indices = np.array([1, 2, 3, 4])

    assert 1 == pytest.approx(iou(pred_indices, gt_indices))


def test_assert_iou_exception(clean_env):
    with pytest.raises(AssertionError) as excinfo:
        pred_labels = np.array([])
        gt_labels = np.array([])

        iou(pred_labels, gt_labels)

    assert (
        str(excinfo.value) == "Prediction and ground truth array sizes must be positive"
    )


def test_iou_real_data(clean_env):
    pred_labels = np.load("tests/data/pred_0.npy")
    gt_labels = np.load("tests/data/gt_0.npy")
    pred = __group_indices_by_labels(pred_labels)
    gt = __group_indices_by_labels(gt_labels)

    assert 0.57 == pytest.approx(iou(pred[0.0], gt[0.0]), 0.01)
