import numpy as np
import pytest
import open3d as o3d

from evops.metrics import mean, iou


def test_mean_simple_array():
    pred_labels = np.array([1, 1, 1])
    gt_labels = np.array([1, 1, 1])

    metric = iou

    assert 1.0 == pytest.approx(mean(pred_labels, gt_labels, metric))


def test_mean_half_result():
    pred_labels = np.array([1, 2, 1, 2])
    gt_labels = np.array([1, 1, 1, 1])

    metric = iou

    assert 0.5 == pytest.approx(mean(pred_labels, gt_labels, metric))


def test_mean_null_result():
    pred_labels = np.array([0, 0, 0, 0])
    gt_labels = np.array([1, 1, 1, 1])

    metric = iou

    assert 0 == pytest.approx(mean(pred_labels, gt_labels, metric))


def test_mean_pred_labels_assert():
    with pytest.raises(AssertionError) as excinfo:
        pred_labels = np.ones((3, 3, 3))
        gt_labels = np.array([1])

        metric = iou
        mean(pred_labels, gt_labels, metric)

    assert str(excinfo.value) == "Incorrect predicted label array size, expected (n)"


def test_mean_gt_labels_assert():
    with pytest.raises(AssertionError) as excinfo:
        pred_labels = np.array([1])
        gt_labels = np.ones((3, 3, 3))

        metric = iou
        mean(pred_labels, gt_labels, metric)

    assert str(excinfo.value) == "Incorrect ground truth label array size, expected (n)"


def test_mean_iou_real_data():
    pred_labels = np.load("tests/data/pred_0.npy")
    gt_labels = np.load("tests/data/gt_0.npy")

    assert 0.87 == pytest.approx(mean(pred_labels, gt_labels, iou), 0.01)
