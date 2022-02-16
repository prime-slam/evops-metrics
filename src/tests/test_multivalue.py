import os

import numpy as np
import open3d as o3d
import pytest

from src.metrics.metrics import multi_value


def test_multi_value_iou_real_data():
    point_cloud = o3d.io.read_point_cloud("data/0.pcd")
    point_cloud = np.asarray(point_cloud.points)
    pred_labels = np.load("data/pred_0.npy")
    gt_labels = np.load("data/gt_0.npy")

    result = multi_value(point_cloud, pred_labels, gt_labels)

    assert 0.8 == pytest.approx(result["precision"], 0.01)
    assert 0.235 == pytest.approx(result["recall"], 0.01)
    assert 0 == pytest.approx(result["under_segmented"], 0.01)
    assert 0 == pytest.approx(result["over_segmented"], 0.01)
    assert 0.76 == pytest.approx(result["missed"], 0.01)
    assert 0.2 == pytest.approx(result["noise"], 0.01)
