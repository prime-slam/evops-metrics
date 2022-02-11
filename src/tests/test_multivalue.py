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

    precision, recall, under_segmented, over_segmented, missed, noise = multi_value(
        point_cloud, pred_labels, gt_labels
    )

    assert 0.67 == pytest.approx(precision, 0.01)
    assert 0.22 == pytest.approx(recall, 0.01)
    assert 0 == pytest.approx(under_segmented, 0.01)
    assert 0 == pytest.approx(over_segmented, 0.01)
    assert 0.78 == pytest.approx(missed, 0.01)
    assert 0.33 == pytest.approx(noise, 0.01)
