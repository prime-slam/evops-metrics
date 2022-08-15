import pytest
from evops.assoc_metrics import quantitative_points


def test_full_points_result():
    results = {(1, 100): 1, (2, 50): 2, (4, 200): 4}
    prev_planes = {1, 2, 4}

    assert 1 == pytest.approx(quantitative_points(results, prev_planes))


def test_half_points_result():
    results = {(1, 100): 1, (2, 200): 4, (3, 100): 3}
    prev_planes = {1, 2, 3, 4}

    assert 0.5 == pytest.approx(quantitative_points(results, prev_planes))


def test_full_points_result_with_none():
    results = {(1, 10): 1, (2, 30): None, (3, 50): 3}
    prev_planes = {1, 3}

    assert 1 == pytest.approx(quantitative_points(results, prev_planes))


def test_none_points_result():
    results = {(1, 10): 3, (2, 5): None, (3, 1000): 1}
    prev_planes = {1, 2, 3}

    assert 0 == pytest.approx(quantitative_points(results, prev_planes))
