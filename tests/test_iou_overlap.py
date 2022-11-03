import numpy as np
import pytest

import evops.metrics.constants
from evops.utils.iou_overlap import __is_overlapped_iou

from fixtures import clean_env


def test_full_iou_overlap(clean_env):
    pred_indices = np.array([1, 2, 3, 4])
    gt_indices = np.array([1, 2, 3, 4])

    assert __is_overlapped_iou(pred_indices, gt_indices)


def test_null_iou_overlap(clean_env):
    pred_indices = np.array([1, 2, 3, 4])
    gt_indices = np.array([5, 6, 7, 8])

    assert not __is_overlapped_iou(pred_indices, gt_indices)


def test_iou_overlap(clean_env):
    evops.metrics.constants.IOU_THRESHOLD_FULL = 0.25

    pred_indices = np.array([1, 2, 3, 4])
    gt_indices = np.array([1, 2, 5, 6])

    assert __is_overlapped_iou(pred_indices, gt_indices)
