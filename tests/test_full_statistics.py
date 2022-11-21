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
import evops
import numpy as np
import pytest

from evops.metrics import full_statistics, iou
from fixtures import clean_env


def test_full_statistics_benchmark_real_data(clean_env):
    pred_labels = np.load("tests/data/pred_0.npy")
    gt_labels = np.load("tests/data/gt_0.npy")

    tp_condition = "iou"
    mean_func = iou

    result = full_statistics(pred_labels, gt_labels, mean_func, tp_condition)

    assert 0.36 == pytest.approx(result["panoptic"], 0.01)
    assert 0.8 == pytest.approx(result["precision"], 0.01)
    assert 0.235 == pytest.approx(result["recall"], 0.01)
    assert 0.36 == pytest.approx(result["fScore"], 0.01)
    assert 0.2 == pytest.approx(result["usr"], 0.01)
    assert 0.0 == pytest.approx(result["osr"], 0.01)
    assert 0.2 == pytest.approx(result["noise"], 0.01)
    assert 0.76 == pytest.approx(result["missed"], 0.01)
    assert 0.99 == pytest.approx(result["mean"], 0.01)


def test_full_statistics_benchmark(clean_env):
    evops.metrics.constants.IOU_THRESHOLD_FULL = 0.5
    pred_labels = np.array([1, 1, 1, 1])
    gt_labels = np.array([1, 1, 2, 4])

    tp_condition = "iou"
    mean_func = iou

    result = full_statistics(pred_labels, gt_labels, mean_func, tp_condition)

    assert 0.25 == pytest.approx(result["panoptic"], 0.01)
    assert 1 == pytest.approx(result["precision"], 0.01)
    assert 0.33 == pytest.approx(result["recall"], 0.01)
    assert 0.5 == pytest.approx(result["fScore"], 0.01)
    assert 1 == pytest.approx(result["usr"], 0.01)
    assert 0 == pytest.approx(result["osr"], 0.01)
    assert 0 == pytest.approx(result["noise"], 0.01)
    assert 0.66 == pytest.approx(result["missed"], 0.01)
    assert 0.5 == pytest.approx(result["mean"], 0.01)


def test_full_statistics_null_benchmark(clean_env):
    pred_labels = np.array([0, 0, 0, 0])
    gt_labels = np.array([1, 0, 1, 0])

    tp_condition = "iou"
    mean_func = iou

    result = full_statistics(pred_labels, gt_labels, mean_func, tp_condition)

    assert 0 == pytest.approx(result["panoptic"], 0.01)
    assert 0 == pytest.approx(result["precision"], 0.01)
    assert 0 == pytest.approx(result["recall"], 0.01)
    assert 0 == pytest.approx(result["fScore"], 0.01)
    assert 0 == pytest.approx(result["usr"], 0.01)
    assert 0 == pytest.approx(result["osr"], 0.01)
    assert 0 == pytest.approx(result["noise"], 0.01)
    assert 1 == pytest.approx(result["missed"], 0.01)
    assert 0 == pytest.approx(result["mean"], 0.01)
