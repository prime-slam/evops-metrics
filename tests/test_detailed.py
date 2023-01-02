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
from evops.metrics import usr, osr, noise, missed

from fixtures import clean_env


def test_usr_benchmark_real_data(clean_env):
    pred_labels = np.load("tests/data/pred_0.npy")
    gt_labels = np.load("tests/data/gt_0.npy")

    tp_condition = "iou"
    result = usr(pred_labels, gt_labels, tp_condition)

    assert 0.2 == pytest.approx(result, 0.01)


def test_osr_benchmark_real_data(clean_env):
    pred_labels = np.load("tests/data/pred_0.npy")
    gt_labels = np.load("tests/data/gt_0.npy")

    tp_condition = "iou"
    result = osr(pred_labels, gt_labels, tp_condition)

    assert 0.0 == pytest.approx(result, 0.01)


def test_noise_benchmark_real_data(clean_env):
    pred_labels = np.load("tests/data/pred_0.npy")
    gt_labels = np.load("tests/data/gt_0.npy")

    tp_condition = "iou"
    result = noise(pred_labels, gt_labels, tp_condition)

    assert 0.2 == pytest.approx(result, 0.01)


def test_missed_benchmark_real_data(clean_env):
    pred_labels = np.load("tests/data/pred_0.npy")
    gt_labels = np.load("tests/data/gt_0.npy")

    tp_condition = "iou"
    result = missed(pred_labels, gt_labels, tp_condition)

    assert 0.76 == pytest.approx(result, 0.01)


def test_usr_benchmark(clean_env):
    evops.metrics.constants.IOU_THRESHOLD_FULL = 0.5
    pred_labels = np.array([1, 1, 1, 1])
    gt_labels = np.array([1, 1, 2, 4])

    tp_condition = "iou"
    result = usr(pred_labels, gt_labels, tp_condition)
    assert 1 == pytest.approx(result, 0.01)


def test_osr_benchmark(clean_env):
    evops.metrics.constants.IOU_THRESHOLD_FULL = 0.5
    pred_labels = np.array([1, 1, 1, 1])
    gt_labels = np.array([1, 1, 2, 4])

    tp_condition = "iou"
    result = osr(pred_labels, gt_labels, tp_condition)
    assert 0 == pytest.approx(result, 0.01)


def test_noise_benchmark(clean_env):
    evops.metrics.constants.IOU_THRESHOLD_FULL = 0.5
    pred_labels = np.array([1, 1, 1, 1])
    gt_labels = np.array([1, 1, 2, 4])

    tp_condition = "iou"
    result = noise(pred_labels, gt_labels, tp_condition)
    assert 0 == pytest.approx(result, 0.01)


def test_missed_benchmark(clean_env):
    evops.metrics.constants.IOU_THRESHOLD_FULL = 0.5
    pred_labels = np.array([1, 1, 1, 1])
    gt_labels = np.array([1, 1, 2, 4])

    tp_condition = "iou"
    result = missed(pred_labels, gt_labels, tp_condition)
    assert 0.66 == pytest.approx(result, 0.01)


def test_usr_null_benchmark(clean_env):
    pred_labels = np.array([0, 0, 0, 0])
    gt_labels = np.array([0, 0, 0, 0])

    tp_condition = "iou"
    result = usr(pred_labels, gt_labels, tp_condition)

    assert 0 == pytest.approx(result, 0.01)


def test_osr_null_benchmark(clean_env):
    pred_labels = np.array([0, 0, 0, 0])
    gt_labels = np.array([0, 0, 0, 0])

    tp_condition = "iou"
    result = osr(pred_labels, gt_labels, tp_condition)

    assert 0 == pytest.approx(result, 0.01)


def test_noise_null_benchmark(clean_env):
    pred_labels = np.array([0, 0, 0, 0])
    gt_labels = np.array([0, 0, 0, 0])

    tp_condition = "iou"
    result = noise(pred_labels, gt_labels, tp_condition)

    assert 0 == pytest.approx(result, 0.01)


def test_missed_null_benchmark(clean_env):
    pred_labels = np.array([0, 0, 0, 0])
    gt_labels = np.array([0, 0, 0, 0])

    tp_condition = "iou"
    result = missed(pred_labels, gt_labels, tp_condition)

    assert 0 == pytest.approx(result, 0.01)
