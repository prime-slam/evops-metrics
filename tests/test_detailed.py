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
