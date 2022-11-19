import numpy as np
import pytest
from evops.metrics.association import (
    quantitative_planes,
    quantitative_planes_with_matching,
)

from fixtures import clean_env


def test_full_planes_result(clean_env):
    pred = {1: 2, 2: 1, 4: 4, 5: 5}
    gt = {1: 2, 2: 1, 4: 4, 5: 5}

    assert 1 == pytest.approx(quantitative_planes(pred, gt))


def test_half_planes_result(clean_env):
    pred = {1: 1, 2: 4, 4: 2, 5: 5}
    gt = {1: 1, 2: 2, 4: 4, 5: 5}

    assert 0.5 == pytest.approx(quantitative_planes(pred, gt))


def test_full_planes_result_with_none(clean_env):
    pred = {1: 1, 2: None, 3: 3}
    gt = {1: 1, 2: None, 3: 3}

    assert 1 == pytest.approx(quantitative_planes(pred, gt))


def test_none_planes_result(clean_env):
    pred = {1: 1, 2: None, 3: 3}
    gt = {1: 3, 2: 4, 3: 1}

    assert 0 == pytest.approx(quantitative_planes(pred, gt))


def test_full_planes_with_matching(clean_env):
    pred = {2: 1, 1: 2, 3: 3}
    pred_cur = np.asarray([0, 2, 0, 1, 1, 3])
    pred_prev = np.asarray([1, 0, 2, 2, 3, 3])
    gt_cur = np.asarray([0, 3, 0, 1, 1, 2])
    gt_prev = np.asarray([3, 0, 1, 1, 2, 2])

    assert 1 == pytest.approx(
        quantitative_planes_with_matching(pred, pred_cur, pred_prev, gt_cur, gt_prev)
    )


def test_removed_planes_with_matching(clean_env):
    pred = {4: None, 2: 1, 1: 2, 3: 3}
    pred_cur = np.asarray([4, 2, 0, 1, 1, 3])
    pred_prev = np.asarray([1, 0, 2, 2, 3, 3])
    gt_cur = np.asarray([6, 3, 0, 1, 1, 2])
    gt_prev = np.asarray([3, 0, 1, 1, 2, 2])

    assert 1 == pytest.approx(
        quantitative_planes_with_matching(pred, pred_cur, pred_prev, gt_cur, gt_prev)
    )


def test_wrong_detected_planes_with_matching_influence_metrics(clean_env):
    pred = {4: 4, 2: 1, 1: 2, 3: 3}
    pred_cur = np.asarray([4, 0, 2, 0, 1, 1, 3])
    pred_prev = np.asarray([4, 1, 0, 2, 2, 3, 3])
    gt_cur = np.asarray([0, 0, 3, 0, 1, 1, 2])
    gt_prev = np.asarray([0, 3, 0, 1, 1, 2, 2])

    assert 0.75 == pytest.approx(
        quantitative_planes_with_matching(pred, pred_cur, pred_prev, gt_cur, gt_prev)
    )


def test_wrong_not_detected_planes_with_matching(clean_env):
    pred = {2: 1, 1: 2, 3: 3}
    pred_cur = np.asarray([0, 2, 0, 1, 1, 3])
    pred_prev = np.asarray([1, 0, 2, 2, 3, 3])
    gt_cur = np.asarray([6, 3, 0, 1, 1, 2])
    gt_prev = np.asarray([3, 0, 1, 1, 2, 2])

    assert 1 == pytest.approx(
        quantitative_planes_with_matching(pred, pred_cur, pred_prev, gt_cur, gt_prev)
    )
