from typing import Any, Callable

import numpy as np
from nptyping import NDArray

from src.utils.metrics_utils import __group_indices_by_labels


def __mean(
    pc_points: NDArray[(Any, 3), np.float64],
    pred_labels: NDArray[Any, np.int32],
    gt_labels: NDArray[Any, np.int32],
    metric: Callable[
        [NDArray[(Any, 3), np.float64], NDArray[Any, np.int32], NDArray[Any, np.int32]],
        np.float64,
    ],
) -> np.float64:
    plane_predicted_dict = __group_indices_by_labels(pred_labels)
    plane_gt_dict = __group_indices_by_labels(gt_labels)
    unique_labels = np.unique(pred_labels)
    mean_array = np.empty((1, 0), np.float64)

    for label in unique_labels:
        if label in plane_gt_dict:
            mean_array = np.append(
                mean_array,
                metric(pc_points, plane_predicted_dict[label], plane_gt_dict[label]),
            )

    if mean_array.size == 0:
        return 0

    return mean_array.mean()
