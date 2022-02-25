from typing import Any
from nptyping import NDArray

import numpy as np


def __dice(
    pc_points: NDArray[(Any, 3), np.float64],
    pred_indices: NDArray[Any, np.int32],
    gt_indices: NDArray[Any, np.int32],
) -> np.float64:
    intersection = np.intersect1d(pred_indices, gt_indices)

    return 2 * intersection.size / (pred_indices.size + gt_indices.size)
