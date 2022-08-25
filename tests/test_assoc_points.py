import pytest
from evops.assoc_metrics import quantitative_points


def test_full_points_result():
    results = {1: 2, 2: 1, 4: 4}
    lengths = {1: 100, 2: 50, 4: 200}
    groundtruth = {1: 2, 2: 1, 4: 4}

    assert 1 == pytest.approx(quantitative_points(results, lengths, groundtruth))


def test_half_points_result():
    results = {1: 2, 2: 1, 3: 5}
    lengths = {1: 100, 2: 200, 3: 100}
    groundtruth = {1: 2, 2: 3, 3: 5}

    assert 0.5 == pytest.approx(quantitative_points(results, lengths, groundtruth))


def test_full_points_result_with_none():
    results = {1: 1, 2: None, 3: 3}
    lengths = {1: 10, 2: 30, 3: 50}
    groundtruth = {1: 1, 2: None, 3: 3}

    assert 1 == pytest.approx(quantitative_points(results, lengths, groundtruth))


def test_none_points_result():
    results = {1: 1, 2: None, 3: 3}
    lengths = {1: 10, 2: 5, 3: 1000}
    groundtruth = {1: 3, 2: 5, 3: 1}

    assert 0 == pytest.approx(quantitative_points(results, lengths, groundtruth))
