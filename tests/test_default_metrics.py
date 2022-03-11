import numpy as np
import pytest
import open3d as o3d

from evops.metrics import (
    precision,
    accuracy,
    recall,
    fScore,
)
from evops.utils.metrics_utils import __group_indices_by_labels


def test_full_precision_result():
    pc_points = np.eye(4, 3)
    pred_labels = np.array([1, 2, 3, 4])
    gt_labels = np.array([1, 2, 3, 4])

    assert 1 == pytest.approx(precision(pc_points, pred_labels, gt_labels))


def test_null_precision_result():
    pc_points = np.eye(4, 3)
    pred_labels = np.array([1, 2, 3, 4])
    gt_labels = np.array([5, 6, 7, 8])

    assert 0 == pytest.approx(precision(pc_points, pred_labels, gt_labels))


def test_half_precision_result():
    pc_points = np.eye(4, 3)
    pred_labels = np.array([1, 2, 3, 4])
    gt_labels = np.array([1, 2, 5, 6])

    assert 0.5 == pytest.approx(precision(pc_points, pred_labels, gt_labels))


def test_full_accuracy_result():
    pc_points = np.eye(4, 3)
    pred_labels = np.array([1, 2, 3, 4])
    gt_labels = np.array([1, 2, 3, 4])

    assert 1 == pytest.approx(accuracy(pc_points, pred_labels, gt_labels))


def test_null_recall_result():
    pc_points = np.eye(4, 3)
    pred_labels = np.array([1, 2, 3, 4])
    gt_labels = np.array([5, 6, 7, 8])

    assert 0 == pytest.approx(recall(pc_points, pred_labels, gt_labels))


def test_full_recall_result():
    pc_points = np.eye(4, 3)
    pred_labels = np.array([1, 2, 3, 4])
    gt_labels = np.array([1, 2, 3, 4])

    assert 1 == pytest.approx(recall(pc_points, pred_labels, gt_labels))


def test_recall_result():
    pc_points = np.eye(4, 3)
    pred_labels = np.array([1, 2, 3, 4])
    gt_labels = np.array([5, 2, 3, 4])

    assert 0.75 == pytest.approx(recall(pc_points, pred_labels, gt_labels))


def test_null_fScore_result():
    pc_points = np.eye(4, 3)
    pred_labels = np.array([1, 2, 3, 4])
    gt_labels = np.array([5, 6, 7, 8])

    assert 0 == pytest.approx(fScore(pc_points, pred_labels, gt_labels))


def test_fScore_result():
    pc_points = np.eye(4, 3)
    pred_labels = np.array([1, 2, 9, 4])
    gt_labels = np.array([5, 2, 7, 8])

    assert 0.25 == pytest.approx(fScore(pc_points, pred_labels, gt_labels))


def test_precision_real_data():
    point_cloud = o3d.io.read_point_cloud("tests/data/0.pcd")
    point_cloud = np.asarray(point_cloud.points)
    pred_labels = np.load("tests/data/pred_0.npy")
    gt_labels = np.load("tests/data/gt_0.npy")

    assert 0.11 == pytest.approx(precision(point_cloud, pred_labels, gt_labels), 0.01)


def test_accuracy_real_data():
    point_cloud = o3d.io.read_point_cloud("tests/data/0.pcd")
    point_cloud = np.asarray(point_cloud.points)
    pred_labels = np.load("tests/data/pred_0.npy")
    gt_labels = np.load("tests/data/gt_0.npy")

    assert 0.11 == pytest.approx(accuracy(point_cloud, pred_labels, gt_labels), 0.01)


def test_recall_real_data():
    point_cloud = o3d.io.read_point_cloud("tests/data/0.pcd")
    point_cloud = np.asarray(point_cloud.points)
    pred_labels = np.load("tests/data/pred_0.npy")
    gt_labels = np.load("tests/data/gt_0.npy")

    assert 0.11 == pytest.approx(recall(point_cloud, pred_labels, gt_labels), 0.01)


def test_fScore_real_data():
    point_cloud = o3d.io.read_point_cloud("tests/data/0.pcd")
    point_cloud = np.asarray(point_cloud.points)
    pred_labels = np.load("tests/data/pred_0.npy")
    gt_labels = np.load("tests/data/gt_0.npy")

    assert 0.11 == pytest.approx(fScore(point_cloud, pred_labels, gt_labels), 0.01)
