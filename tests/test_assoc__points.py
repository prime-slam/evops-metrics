import pytest
from evops.assoc_metrics import quantitative_points


def test_full_points_result():
    results = {(1, 100): 2, (2, 50): 1, (4, 200): 4}
    groundtruth = {1: 2, 2: 1, 4: 4}

    assert 1 == pytest.approx(quantitative_points(results, groundtruth))


def test_half_points_result():
    results = {(1, 100): 2, (2, 200): 1, (3, 100): 5}
    groundtruth = {1: 2, 2: 3, 3: 5}

    assert 0.5 == pytest.approx(quantitative_points(results, groundtruth))


def test_full_points_result_with_none():
    results = {(1, 10): 1, (2, 30): None, (3, 50): 3}
    groundtruth = {1: 1, 2: None, 3: 3}

    assert 1 == pytest.approx(quantitative_points(results, groundtruth))


def test_none_points_result():
    results = {(1, 10): 1, (2, 5): None, (3, 1000): 3}
    groundtruth = {1: 3, 2: 5, 3: 1}

    assert 0 == pytest.approx(quantitative_points(results, groundtruth))
