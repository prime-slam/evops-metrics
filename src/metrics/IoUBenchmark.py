from typing import Any

import numpy as np
from nptyping import NDArray


def __iou(
    pc_points: NDArray[(Any, 3), np.float64],
    pred_indices: NDArray[Any, np.int32],
    gt_indices: NDArray[Any, np.int32],
) -> np.float64:
    intersection = np.intersect1d(pred_indices, gt_indices)
    union = np.union1d(pred_indices, gt_indices)

    return intersection.size / union.size
