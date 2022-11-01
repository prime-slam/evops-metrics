import numpy as np
import pytest

from evops.metrics import detailed


def test_detailed_benchmark_real_data():
    pred_labels = np.load("tests/data/pred_0.npy")
    gt_labels = np.load("tests/data/gt_0.npy")

    tp_condition = "iou"
    result = detailed(pred_labels, gt_labels, tp_condition)

    assert 0.2 == pytest.approx(result["under_segmented"], 0.01)
    assert 0.0 == pytest.approx(result["over_segmented"], 0.01)
    assert 0.76 == pytest.approx(result["missed"], 0.01)
    assert 0.2 == pytest.approx(result["noise"], 0.01)


def test_detailed_benchmark():
    pred_labels = np.array([1, 1, 1, 1])
    gt_labels = np.array([1, 1, 2, 4])

    tp_condition = "iou"
    result = detailed(pred_labels, gt_labels, tp_condition)
    assert 1 == pytest.approx(result["under_segmented"], 0.01)


def test_detailed_null_benchmark():
    pred_labels = np.array([0, 0, 0, 0])
    gt_labels = np.array([0, 0, 0, 0])

    tp_condition = "iou"
    result = detailed(pred_labels, gt_labels, tp_condition)

    assert 0 == pytest.approx(result["under_segmented"], 0.01)
    assert 0 == pytest.approx(result["over_segmented"], 0.01)
    assert 0 == pytest.approx(result["missed"], 0.01)
    assert 0 == pytest.approx(result["noise"], 0.01)
