import numpy as np
import pytest
import open3d as o3d

from evops.metrics import iou
from evops.utils.MetricsUtils import __group_indices_by_labels


def test_null_iou_result():
    pred_indices = np.array([1, 2, 3, 4])
    gt_indices = np.array([5, 6, 7, 8])

    assert 0 == pytest.approx(iou(pred_indices, gt_indices))


def test_full_iou_result():
    pred_indices = np.array([1, 2, 3, 4])
    gt_indices = np.array([1, 2, 3, 4])

    assert 1 == pytest.approx(iou(pred_indices, gt_indices))


def test_assert_iou_exception():
    with pytest.raises(AssertionError) as excinfo:
        pred_labels = np.array([])
        gt_labels = np.array([])

        iou(pred_labels, gt_labels)

    assert str(excinfo.value) == "Array sizes must be positive"


def test_iou_real_data():
    pred_labels = np.load("tests/data/pred_0.npy")
    gt_labels = np.load("tests/data/gt_0.npy")
    pred = __group_indices_by_labels(pred_labels)
    gt = __group_indices_by_labels(gt_labels)

    assert 0.57 == pytest.approx(iou(pred[0.0], gt[0.0]), 0.01)
