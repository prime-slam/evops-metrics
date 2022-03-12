import numpy as np
import open3d as o3d
import pytest

from evops.metrics import multi_value


def test_multi_value_benchmark_real_data():
    pred_labels = np.load("tests/data/pred_0.npy")
    gt_labels = np.load("tests/data/gt_0.npy")

    result = multi_value(pred_labels, gt_labels)

    assert 0.8 == pytest.approx(result["precision"], 0.01)
    assert 0.235 == pytest.approx(result["recall"], 0.01)
    assert 0.8 == pytest.approx(result["under_segmented"], 0.01)
    assert 0.176 == pytest.approx(result["over_segmented"], 0.01)
    assert 0.76 == pytest.approx(result["missed"], 0.01)
    assert 0.2 == pytest.approx(result["noise"], 0.01)


def test_multi_value_benchmark():
    pred_labels = np.array([1, 1, 1, 1])
    gt_labels = np.array([1, 1, 2, 4])

    result = multi_value(pred_labels, gt_labels)
    assert 1 == pytest.approx(result["under_segmented"], 0.01)


def test_multi_value_null_benchmark():
    pred_labels = np.array([0, 0, 0, 0])
    gt_labels = np.array([0, 0, 0, 0])

    result = multi_value(pred_labels, gt_labels)

    assert 0 == pytest.approx(result["precision"], 0.01)
    assert 0 == pytest.approx(result["recall"], 0.01)
    assert 0 == pytest.approx(result["under_segmented"], 0.01)
    assert 0 == pytest.approx(result["over_segmented"], 0.01)
    assert 0 == pytest.approx(result["missed"], 0.01)
    assert 0 == pytest.approx(result["noise"], 0.01)
