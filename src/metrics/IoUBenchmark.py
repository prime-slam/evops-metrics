import numpy as np
import numpy.typing as npt


def iou(
    pc_points: npt.NDArray[np.float64],
    pred_indices: npt.NDArray[np.float64],
    gt_indices: npt.NDArray[np.float64],
) -> np.float64:
    intersection = np.intersect1d(pred_indices, gt_indices)
    union = np.union1d(pred_indices, gt_indices)
    return intersection.size / union.size
