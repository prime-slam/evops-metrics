import numpy as np
import pytest

from evops.metrics import iou, panoptic


def test_panoptic_simple_array():
    pred_labels = np.array([1, 1, 1])
    gt_labels = np.array([1, 1, 1])

    metric = iou
    tp_condition = "iou"

    assert 1.0 == pytest.approx(panoptic(pred_labels, gt_labels, metric, tp_condition))


def test_panoptic_part_result_void():
    pred_labels = np.array([1, 0, 1, 1, 1])
    gt_labels = np.array([1, 1, 1, 1, 1])

    metric = iou
    tp_condition = "iou"

    assert 0.8 == pytest.approx(panoptic(pred_labels, gt_labels, metric, tp_condition))


def test_panoptic_part_result_other():
    pred_labels = np.array([1, 2, 1, 1, 1])
    gt_labels = np.array([1, 1, 1, 1, 1])

    metric = iou
    tp_condition = "iou"

    assert 0.53 == pytest.approx(
        panoptic(pred_labels, gt_labels, metric, tp_condition), 0.01
    )


def test_panoptic_half_result():
    pred_labels = np.array([1, 2, 1, 2])
    gt_labels = np.array([1, 1, 1, 1])

    metric = iou
    tp_condition = "iou"

    assert 0.0 == pytest.approx(panoptic(pred_labels, gt_labels, metric, tp_condition))


def test_panoptic_null_result():
    pred_labels = np.array([0, 0, 0, 0])
    gt_labels = np.array([1, 1, 1, 1])

    metric = iou
    tp_condition = "iou"

    assert 0 == pytest.approx(panoptic(pred_labels, gt_labels, metric, tp_condition))


def test_panoptic_pred_labels_assert():
    with pytest.raises(AssertionError) as excinfo:
        pred_labels = np.ones((3, 3, 3))
        gt_labels = np.array([1])

        metric = iou
        tp_condition = "iou"
        panoptic(pred_labels, gt_labels, metric, tp_condition)

    assert str(excinfo.value) == "Incorrect predicted label array size, expected (n)"


def test_panoptic_gt_labels_assert():
    with pytest.raises(AssertionError) as excinfo:
        pred_labels = np.array([1])
        gt_labels = np.ones((3, 3, 3))

        metric = iou
        tp_condition = "iou"
        panoptic(pred_labels, gt_labels, metric, tp_condition)

    assert str(excinfo.value) == "Incorrect ground truth label array size, expected (n)"


def test_panoptic_iou_real_data():
    pred_labels = np.load("tests/data/pred_0.npy")
    gt_labels = np.load("tests/data/gt_0.npy")

    metric = iou
    tp_condition = "iou"

    assert 0.36 == pytest.approx(
        panoptic(pred_labels, gt_labels, metric, tp_condition), 0.01
    )
