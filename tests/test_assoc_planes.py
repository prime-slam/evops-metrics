import pytest
from evops.assoc_metrics import quantitative_planes


def test_full_planes_result():
    results = {1: 1, 2: 2, 4: 4, 5: 5}
    prev_planes = {1, 2, 4, 5}

    assert 1 == pytest.approx(quantitative_planes(results, prev_planes))


def test_half_planes_result():
    results = {1: 1, 2: 4, 4: 2, 5: 5}
    prev_planes = {1, 2, 4, 5}

    assert 0.5 == pytest.approx(quantitative_planes(results, prev_planes))


def test_full_planes_result_with_none():
    results = {1: 1, 2: None, 3: 3}
    prev_planes = {1, 3}

    assert 1 == pytest.approx(quantitative_planes(results, prev_planes))


def test_none_planes_result():
    results = {1: 3, 2: None, 3: 1}
    prev_planes = {1, 2, 3}

    assert 0 == pytest.approx(quantitative_planes(results, prev_planes))
