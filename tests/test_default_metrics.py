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
from evops.metrics import (
    precision,
    recall,
    fScore,
)
from fixtures import clean_env


def test_full_precision_iou_statistics_result(clean_env):
    tp_condition = "iou"
    pred_labels = np.array([1, 2, 3, 4])
    gt_labels = np.array([1, 2, 3, 4])

    assert 1 == pytest.approx(precision(pred_labels, gt_labels, tp_condition))


def test_half_precision_iou_statistics_result(clean_env):
    tp_condition = "iou"
    pred_labels = np.array([1, 1, 3, 3])
    gt_labels = np.array([2, 2, 0, 3])

    assert 0.5 == pytest.approx(precision(pred_labels, gt_labels, tp_condition))


def test_precision_iou_statistics_empty_prediction(clean_env):
    tp_condition = "iou"
    pred_labels = np.array([0, 0, 0, 0])
    gt_labels = np.array([1, 1, 1, 1])

    assert 0 == precision(pred_labels, gt_labels, tp_condition)


def test_null_recall_iou_statistics_result(clean_env):
    tp_condition = "iou"
    pred_labels = np.array([1, 1, 1, 1])
    gt_labels = np.array([1, 2, 3, 4])

    assert 0 == pytest.approx(recall(pred_labels, gt_labels, tp_condition))


def test_full_recall_iou_statistics_result(clean_env):
    tp_condition = "iou"
    pred_labels = np.array([1, 2, 3, 4])
    gt_labels = np.array([5, 6, 7, 8])

    assert 1 == pytest.approx(recall(pred_labels, gt_labels, tp_condition))


def test_full_recall_with_two_planes_iou_statistics_result(clean_env):
    tp_condition = "iou"
    pred_labels = np.array([1, 1, 2, 2])
    gt_labels = np.array([2, 2, 3, 3])

    assert 1 == pytest.approx(recall(pred_labels, gt_labels, tp_condition))


def test_half_recall_iou_statistics_result(clean_env):
    tp_condition = "iou"
    pred_labels = np.array([2, 2, 0, 3])
    gt_labels = np.array([1, 1, 3, 3])

    assert 0.5 == pytest.approx(recall(pred_labels, gt_labels, tp_condition))


def test_recall_iou_statistics_empty_prediction(clean_env):
    tp_condition = "iou"
    pred_labels = np.array([1, 1, 1, 1])
    gt_labels = np.array([0, 0, 0, 0])

    assert 0 == recall(pred_labels, gt_labels, tp_condition)


def test_full_fScore_iou_statistics_result(clean_env):
    tp_condition = "iou"
    pred_labels = np.array([1, 2, 3, 4])
    gt_labels = np.array([5, 6, 7, 8])

    assert 1 == pytest.approx(fScore(pred_labels, gt_labels, tp_condition))


def test_almost_half_fScore_iou_statistics_result(clean_env):
    tp_condition = "iou"
    pred_labels = np.array([1, 1, 2, 3])
    gt_labels = np.array([4, 5, 6, 7])

    assert 0.57 == pytest.approx(fScore(pred_labels, gt_labels, tp_condition), 0.01)


def test_precision_real_data_iou_statistics(clean_env):
    evops.metrics.constants.IOU_THRESHOLD_FULL = 0.5
    evops.metrics.constants.UNSEGMENTED_LABEL = 0

    tp_condition = "iou"
    pred_labels = np.load("tests/data/pred_0.npy")
    gt_labels = np.load("tests/data/gt_0.npy")

    assert 0.8 == pytest.approx(precision(pred_labels, gt_labels, tp_condition), 0.01)


def test_recall_real_data_iou_statistics(clean_env):
    evops.metrics.constants.IOU_THRESHOLD_FULL = 0.5
    evops.metrics.constants.UNSEGMENTED_LABEL = 0

    tp_condition = "iou"
    pred_labels = np.load("tests/data/pred_0.npy")
    gt_labels = np.load("tests/data/gt_0.npy")

    assert 0.235 == pytest.approx(recall(pred_labels, gt_labels, tp_condition), 0.01)


def test_fScore_real_data_iou_statistics(clean_env):
    evops.metrics.constants.IOU_THRESHOLD_FULL = 0.5
    evops.metrics.constants.UNSEGMENTED_LABEL = 0

    tp_condition = "iou"
    pred_labels = np.load("tests/data/pred_0.npy")
    gt_labels = np.load("tests/data/gt_0.npy")

    assert 0.36 == pytest.approx(fScore(pred_labels, gt_labels, tp_condition), 0.01)
