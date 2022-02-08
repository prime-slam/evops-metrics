import numpy as np
import pytest

from src.metrics.Benchmark import mean
from src.metrics.IoUBenchmark import iou


def test_mean_simple_array():
    pc_points = np.eye(3, 3)
    pred_labels = np.array([1, 1, 1])
    gt_labels = np.array([1, 1, 1])

    metrics = [iou]

    assert [1.0] == pytest.approx(mean(pc_points, pred_labels, gt_labels, metrics))


def test_mean_half_result():
    pc_points = np.eye(4, 3)
    pred_labels = np.array([1, 2, 1, 2])
    gt_labels = np.array([1, 1, 1, 1])

    metrics = [iou]

    assert [0.5] == pytest.approx(mean(pc_points, pred_labels, gt_labels, metrics))


def test_mean_null_result():
    pc_points = np.eye(4, 3)
    pred_labels = np.array([1, 2, 3, 4])
    gt_labels = np.array([5, 6, 7, 8])

    metrics = [iou]

    assert [0] == pytest.approx(mean(pc_points, pred_labels, gt_labels, metrics))


def test_mean_first_assert():
    with pytest.raises(AssertionError) as excinfo:
        pc_points = np.eye(1, 2)
        pred_labels = np.array([])
        gt_labels = np.array([])

        metrics = [iou]
        mean(pc_points, pred_labels, gt_labels, metrics)

    assert str(excinfo.value) == "Dimension of the array of points should be (n, 3)"


def test_mean_second_assert():
    with pytest.raises(AssertionError) as excinfo:
        pc_points = np.eye(1, 3)
        pred_labels = np.array([])
        gt_labels = np.array([1])

        metrics = [iou]
        mean(pc_points, pred_labels, gt_labels, metrics)

    assert (
        str(excinfo.value)
        == "Number of points does not match the array of predicted labels"
    )


def test_mean_third_assert():
    with pytest.raises(AssertionError) as excinfo:
        pc_points = np.eye(1, 3)
        pred_labels = np.array([1])
        gt_labels = np.array([])

        metrics = [iou]
        mean(pc_points, pred_labels, gt_labels, metrics)

    assert (
        str(excinfo.value)
        == "Number of points does not match the array of ground truth labels"
    )
