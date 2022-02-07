import numpy as np
import pytest

from src.metrics.one_value.IoUBenchmark import IoUBenchmark


def test_cumulative_simple_array():
    point_cloud = np.empty((0, 3), np.float64)
    point_cloud_predicted = np.array([1, 1, 1])
    point_cloud_gt = np.array([1, 1, 1])

    iouBenchmark = IoUBenchmark()

    assert 1.0 == pytest.approx(
        iouBenchmark.calculate_cumulative_metric(
            point_cloud, point_cloud_predicted, point_cloud_gt
        )
    )


def test_cumulative_half_iou_result():
    point_cloud = np.empty((0, 3), np.float64)
    point_cloud_predicted = np.array([1, 2, 1, 2])
    point_cloud_gt = np.array([1, 1, 1, 1])

    iouBenchmark = IoUBenchmark()

    assert 0.5 == pytest.approx(
        iouBenchmark.calculate_cumulative_metric(
            point_cloud, point_cloud_predicted, point_cloud_gt
        )
    )


def test_cumulative_null_iou_result():
    point_cloud = np.empty((0, 3), np.float64)
    point_cloud_predicted = np.array([1, 2, 3, 4])
    point_cloud_gt = np.array([5, 6, 7, 8])

    iouBenchmark = IoUBenchmark()

    assert 0 == pytest.approx(
        iouBenchmark.calculate_cumulative_metric(
            point_cloud, point_cloud_predicted, point_cloud_gt
        )
    )
