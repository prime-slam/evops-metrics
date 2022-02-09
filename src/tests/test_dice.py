import numpy as np
import pytest

from src.metrics.DiceBenchmark import dice


def test_assert_dice_exception():
    with pytest.raises(AssertionError) as excinfo:
        pc_points = np.eye(1, 3)
        pred_labels = np.array([])
        gt_labels = np.array([])

        dice(pc_points, pred_labels, gt_labels)

    assert str(excinfo.value) == "Array sizes must be positive"


def test_null_dice_result():
    pc_points = np.eye(4, 3)
    pred_indices = np.array([1, 2, 3, 4])
    gt_indices = np.array([5, 6, 7, 8])

    assert 0 == pytest.approx(dice(pc_points, pred_indices, gt_indices))


def test_full_dice_result():
    pc_points = np.eye(4, 3)
    pred_indices = np.array([1, 2, 3, 4])
    gt_indices = np.array([1, 2, 3, 4])

    assert 1 == pytest.approx(dice(pc_points, pred_indices, gt_indices))


def test_half_dice_result():
    pc_points = np.eye(4, 3)
    pred_indices = np.array([1, 2, 3, 4])
    gt_indices = np.array([1, 2, 5, 6])

    assert 0.5 == pytest.approx(dice(pc_points, pred_indices, gt_indices))
