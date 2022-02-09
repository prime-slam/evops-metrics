import numpy as np
import pytest

from src.metrics.MultiValueBenchmark import (
    precision,
    accuracy,
    recall,
    fScore,
    get_all_multi_value_metrics,
)


def test_full_precision_result():
    pc_points = np.eye(4, 3)
    pred_indices = np.array([1, 2, 3, 4])
    gt_indices = np.array([1, 2, 3, 4])

    assert 1 == pytest.approx(precision(pc_points, pred_indices, gt_indices))


def test_null_precision_result():
    pc_points = np.eye(4, 3)
    pred_indices = np.array([1, 2, 3, 4])
    gt_indices = np.array([5, 6, 7, 8])

    assert 0 == pytest.approx(precision(pc_points, pred_indices, gt_indices))


def test_half_precision_result():
    pc_points = np.eye(4, 3)
    pred_indices = np.array([1, 2, 3, 4])
    gt_indices = np.array([3, 4, 5, 6])

    assert 0.5 == pytest.approx(precision(pc_points, pred_indices, gt_indices))


def test_full_accuracy_result():
    pc_points = np.eye(4, 3)
    pred_indices = np.array([1, 2, 3, 4])
    gt_indices = np.array([1, 2, 3, 4])

    assert 1 == pytest.approx(accuracy(pc_points, pred_indices, gt_indices))


def test_accuracy_result():
    pc_points = np.eye(4, 3)
    pred_indices = np.array([5, 3, 4, 2])
    gt_indices = np.array([3, 1, 1, 1])

    assert 0.8 == pytest.approx(accuracy(pc_points, pred_indices, gt_indices))


def test_null_recall_result():
    pc_points = np.eye(4, 3)
    pred_indices = np.array([1, 2, 3, 4])
    gt_indices = np.array([5, 6, 7, 8])

    assert 0 == pytest.approx(recall(pc_points, pred_indices, gt_indices))


def test_full_recall_result():
    pc_points = np.eye(4, 3)
    pred_indices = np.array([1, 2, 3, 4])
    gt_indices = np.array([1, 2, 3, 4])

    assert 1 == pytest.approx(recall(pc_points, pred_indices, gt_indices))


def test_recall_result():
    pc_points = np.eye(4, 3)
    pred_indices = np.array([1, 2, 3, 4])
    gt_indices = np.array([5, 2, 3, 4])

    assert 0.75 == pytest.approx(recall(pc_points, pred_indices, gt_indices))


def test_null_fScore_result():
    pc_points = np.eye(4, 3)
    pred_indices = np.array([1, 2, 3, 4])
    gt_indices = np.array([5, 6, 7, 8])

    assert 0 == pytest.approx(fScore(pc_points, pred_indices, gt_indices))


def test_fScore_result():
    pc_points = np.eye(4, 3)
    pred_indices = np.array([1, 2, 9, 4])
    gt_indices = np.array([5, 2, 7, 8])

    assert 0.25 == pytest.approx(fScore(pc_points, pred_indices, gt_indices))


def test_get_all_multi_value_metrics_result():
    pc_points = np.eye(9, 3)
    pred_labels = np.array([1, 2, 3, 4, 1, 1, 2, 3, 4])
    gt_labels = np.array([1, 2, 3, 4, 3, 3, 3, 2, 2])

    all_metric = get_all_multi_value_metrics(pc_points, pred_labels, gt_labels)

    np.testing.assert_array_almost_equal(all_metric[1], [0.33, 0.92, 1, 0.5], 2)
    np.testing.assert_array_almost_equal(all_metric[2], [0.5, 0.89, 0.33, 0.4], 2)
    np.testing.assert_array_almost_equal(all_metric[3], [0.5, 0.85, 0.25, 0.33], 2)
    np.testing.assert_array_almost_equal(all_metric[4], [0.5, 0.96, 1, 0.67], 2)
