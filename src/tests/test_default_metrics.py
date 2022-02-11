import numpy as np
import pytest

from src.metrics.metrics import (
    precision,
    accuracy,
    recall,
    fScore,
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
