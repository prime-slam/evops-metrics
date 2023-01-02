# Copyright (c) 2022, Pavel Mokeev, Dmitrii Iarosh, Anastasiia Kornilova, Ivan Moskalenko
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
import numpy as np
import pytest
from evops.metrics.association import (
    matched_points_ratio,
    matched_points_ratio_with_matching,
)

from fixtures import clean_env


def test_full_points_result(clean_env):
    pred = {1: 2, 2: 1, 4: 4}
    sizes = {1: 100, 2: 50, 4: 200}
    gt = {1: 2, 2: 1, 4: 4}

    assert 1 == pytest.approx(matched_points_ratio(pred, sizes, gt))


def test_half_points_result(clean_env):
    pred = {1: 2, 2: 1, 3: 5}
    sizes = {1: 100, 2: 200, 3: 100}
    gt = {1: 2, 2: 3, 3: 5}

    assert 0.5 == pytest.approx(matched_points_ratio(pred, sizes, gt))


def test_full_points_result_with_none(clean_env):
    pred = {1: 1, 2: None, 3: 3}
    sizes = {1: 10, 2: 30, 3: 50}
    gt = {1: 1, 2: None, 3: 3}

    assert 1 == pytest.approx(matched_points_ratio(pred, sizes, gt))


def test_none_points_result(clean_env):
    pred = {1: 1, 2: None, 3: 3}
    sizes = {1: 10, 2: 5, 3: 1000}
    gt = {1: 3, 2: 5, 3: 1}

    assert 0 == pytest.approx(matched_points_ratio(pred, sizes, gt))


def test_full_planes_with_matching(clean_env):
    pred = {2: 1, 1: 2, 3: 3}
    pred_cur = np.asarray([0, 2, 0, 1, 1, 3])
    pred_prev = np.asarray([1, 0, 2, 2, 3, 3])
    gt_cur = np.asarray([0, 3, 0, 1, 1, 2])
    gt_prev = np.asarray([3, 0, 1, 1, 2, 2])

    assert 1 == pytest.approx(
        matched_points_ratio_with_matching(pred, pred_cur, pred_prev, gt_cur, gt_prev)
    )


def test_removed_planes_with_matching(clean_env):
    pred = {4: None, 2: 1, 1: 2, 3: 3}
    pred_cur = np.asarray([4, 2, 0, 1, 1, 3])
    pred_prev = np.asarray([1, 0, 2, 2, 3, 3])
    gt_cur = np.asarray([6, 3, 0, 1, 1, 2])
    gt_prev = np.asarray([3, 0, 1, 1, 2, 2])

    assert 1 == pytest.approx(
        matched_points_ratio_with_matching(pred, pred_cur, pred_prev, gt_cur, gt_prev)
    )


def test_wrong_detected_planes_with_matching_influence_metrics(clean_env):
    pred = {4: 4, 2: 1, 1: 2, 3: 3}
    pred_cur = np.asarray([4, 0, 2, 0, 1, 1, 3])
    pred_prev = np.asarray([4, 1, 0, 2, 2, 3, 3])
    gt_cur = np.asarray([0, 0, 3, 0, 1, 1, 2])
    gt_prev = np.asarray([0, 3, 0, 1, 1, 2, 2])

    assert 0.8 == pytest.approx(
        matched_points_ratio_with_matching(pred, pred_cur, pred_prev, gt_cur, gt_prev)
    )


def test_wrong_not_detected_planes_with_matching(clean_env):
    pred = {2: 1, 1: 2, 3: 3}
    pred_cur = np.asarray([0, 2, 0, 1, 1, 3])
    pred_prev = np.asarray([1, 0, 2, 2, 3, 3])
    gt_cur = np.asarray([6, 3, 0, 1, 1, 2])
    gt_prev = np.asarray([3, 0, 1, 1, 2, 2])

    assert 1 == pytest.approx(
        matched_points_ratio_with_matching(pred, pred_cur, pred_prev, gt_cur, gt_prev)
    )
