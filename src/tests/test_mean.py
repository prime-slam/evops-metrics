import numpy as np
import pytest

from src.metrics.Benchmark import mean
from src.metrics.IoUBenchmark import iou


def test_mean_simple_array():
    pc_points = np.empty((0, 3), np.float64)
    pred_labels = np.array([1, 1, 1])
    gt_labels = np.array([1, 1, 1])

    metrics = [iou]

    assert [1.0] == pytest.approx(mean(pc_points, pred_labels, gt_labels, metrics))


def test_mean_half_result():
    pc_points = np.empty((0, 3), np.float64)
    pred_labels = np.array([1, 2, 1, 2])
    gt_labels = np.array([1, 1, 1, 1])

    metrics = [iou]

    assert [0.5] == pytest.approx(mean(pc_points, pred_labels, gt_labels, metrics))


def test_cumulative_null_result():
    pc_points = np.empty((0, 3), np.float64)
    pred_labels = np.array([1, 2, 3, 4])
    gt_labels = np.array([5, 6, 7, 8])

    metrics = [iou]

    assert [0] == pytest.approx(mean(pc_points, pred_labels, gt_labels, metrics))
