import numpy as np
import pytest

import evops.metrics.constants
from evops.utils.MetricsUtils import __calc_tp
from fixtures import clean_env


def test_get_tp_using_iou(clean_env):
    pred_labels = np.array([1, 1, 3, 3])
    gt_labels = np.array([2, 2, 0, 3])

    assert 1 == __calc_tp(pred_labels, gt_labels, "iou")


def test_get_tp_null_using_iou(clean_env):
    pred_labels = np.array([1, 2, 3, 4])
    gt_labels = np.array([1, 1, 1, 1])

    assert 0 == __calc_tp(pred_labels, gt_labels, "iou")


def test_get_tp_full_using_iou(clean_env):
    pred_labels = np.array([1, 2, 3, 4])
    gt_labels = np.array([1, 2, 3, 4])

    assert 4 == __calc_tp(pred_labels, gt_labels, "iou")
