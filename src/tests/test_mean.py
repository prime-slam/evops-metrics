import numpy as np
import pytest
import open3d as o3d

from src.metrics.metrics import mean, iou
from src.utils.metrics_utils import __group_indices_by_labels


def test_mean_simple_array():
    pc_points = np.eye(3, 3)
    pred_labels = np.array([1, 1, 1])
    gt_labels = np.array([1, 1, 1])

    metric = iou

    assert 1.0 == pytest.approx(mean(pc_points, pred_labels, gt_labels, metric))


def test_mean_half_result():
    pc_points = np.eye(4, 3)
    pred_labels = np.array([1, 2, 1, 2])
    gt_labels = np.array([1, 1, 1, 1])

    metric = iou

    assert 0.5 == pytest.approx(mean(pc_points, pred_labels, gt_labels, metric))


def test_mean_null_result():
    pc_points = np.eye(4, 3)
    pred_labels = np.array([1, 2, 3, 4])
    gt_labels = np.array([5, 6, 7, 8])

    metric = iou

    assert 0 == pytest.approx(mean(pc_points, pred_labels, gt_labels, metric))


def test_mean_first_assert():
    with pytest.raises(AssertionError) as excinfo:
        pc_points = np.eye(1, 2)
        pred_labels = np.array([])
        gt_labels = np.array([])

        metric = iou
        mean(pc_points, pred_labels, gt_labels, metric)

    assert str(excinfo.value) == "Incorrect point cloud array size, expected (n, 3)"


def test_mean_second_assert():
    with pytest.raises(AssertionError) as excinfo:
        pc_points = np.eye(1, 3)
        pred_labels = np.array([])
        gt_labels = np.array([1])

        metric = iou
        mean(pc_points, pred_labels, gt_labels, metric)

    assert (
        str(excinfo.value)
        == "Number of points does not match the array of predicted labels"
    )


def test_mean_third_assert():
    with pytest.raises(AssertionError) as excinfo:
        pc_points = np.eye(1, 3)
        pred_labels = np.array([1])
        gt_labels = np.array([])

        metric = iou
        mean(pc_points, pred_labels, gt_labels, metric)

    assert (
        str(excinfo.value)
        == "Number of points does not match the array of ground truth labels"
    )


def test_mean_shape_pc_assert():
    with pytest.raises(AssertionError) as excinfo:
        pc_points = np.ones((3, 3, 3))
        pred_labels = np.array([1])
        gt_labels = np.array([])

        metric = iou
        mean(pc_points, pred_labels, gt_labels, metric)

    assert str(excinfo.value) == "Incorrect point cloud array size, expected (n, 3)"


def test_mean_pred_labels_assert():
    with pytest.raises(AssertionError) as excinfo:
        pc_points = np.eye(1, 3)
        pred_labels = np.ones((3, 3, 3))
        gt_labels = np.array([1])

        metric = iou
        mean(pc_points, pred_labels, gt_labels, metric)

    assert str(excinfo.value) == "Incorrect predicted label array size, expected (n)"


def test_mean_gt_labels_assert():
    with pytest.raises(AssertionError) as excinfo:
        pc_points = np.eye(1, 3)
        pred_labels = np.array([1])
        gt_labels = np.ones((3, 3, 3))

        metric = iou
        mean(pc_points, pred_labels, gt_labels, metric)

    assert str(excinfo.value) == "Incorrect ground truth label array size, expected (n)"


def test_mean_iou_real_data():
    point_cloud = o3d.io.read_point_cloud("data/0.pcd")
    point_cloud = np.asarray(point_cloud.points)
    pred_labels = np.load("data/pred_0.npy")
    gt_labels = np.load("data/gt_0.npy")

    assert 0.095 == pytest.approx(mean(point_cloud, pred_labels, gt_labels, iou), 0.01)
