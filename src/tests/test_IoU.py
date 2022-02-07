import numpy as np
import pytest

from src.metrics.one_value.IoUBenchmark import IoUBenchmark


def test_cumulative_simple_array():
    pc_points = np.empty((0, 3), np.float64)
    pred_labels = np.array([1, 1, 1])
    gt_labels = np.array([1, 1, 1])

    iouBenchmark = IoUBenchmark()

    assert 1.0 == pytest.approx(
        iouBenchmark.calculate_metric_mean(pc_points, pred_labels, gt_labels)
    )


def test_cumulative_half_iou_result():
    pc_points = np.empty((0, 3), np.float64)
    pred_labels = np.array([1, 2, 1, 2])
    gt_labels = np.array([1, 1, 1, 1])

    iouBenchmark = IoUBenchmark()

    assert 0.5 == pytest.approx(
        iouBenchmark.calculate_metric_mean(pc_points, pred_labels, gt_labels)
    )


def test_cumulative_null_iou_result():
    pc_points = np.empty((0, 3), np.float64)
    pred_labels = np.array([1, 2, 3, 4])
    gt_labels = np.array([5, 6, 7, 8])

    iouBenchmark = IoUBenchmark()

    assert 0 == pytest.approx(
        iouBenchmark.calculate_metric_mean(pc_points, pred_labels, gt_labels)
    )


def test_null_iou_result():
    pc_points = np.empty((0, 3), np.float64)
    pred_indices = np.array([1, 2, 3, 4])
    gt_indices = np.array([5, 6, 7, 8])

    iouBenchmark = IoUBenchmark()

    assert 0 == pytest.approx(
        iouBenchmark.calculate_metric(pc_points, pred_indices, gt_indices)
    )


def test_full_iou_result():
    pc_points = np.empty((0, 3), np.float64)
    pred_indices = np.array([1, 2, 3, 4])
    gt_indices = np.array([1, 2, 3, 4])

    iouBenchmark = IoUBenchmark()

    assert 1 == pytest.approx(
        iouBenchmark.calculate_metric(pc_points, pred_indices, gt_indices)
    )
