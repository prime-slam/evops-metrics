import pytest
from evops.assoc_metrics import quantitative_planes


def test_full_planes_result():
    results = {1: 2, 2: 1, 4: 4, 5: 5}
    groundtruth = {1: 2, 2: 1, 4: 4, 5: 5}

    assert 1 == pytest.approx(quantitative_planes(results, groundtruth))


def test_half_planes_result():
    results = {1: 1, 2: 4, 4: 2, 5: 5}
    groundtruth = {1: 1, 2: 2, 4: 4, 5: 5}

    assert 0.5 == pytest.approx(quantitative_planes(results, groundtruth))


def test_full_planes_result_with_none():
    results = {1: 1, 2: None, 3: 3}
    groundtruth = {1: 1, 2: None, 3: 3}

    assert 1 == pytest.approx(quantitative_planes(results, groundtruth))


def test_none_planes_result():
    results = {1: 1, 2: None, 3: 3}
    groundtruth = {1: 3, 2: 4, 3: 1}

    assert 0 == pytest.approx(quantitative_planes(results, groundtruth))
