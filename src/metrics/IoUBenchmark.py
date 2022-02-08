import numpy as np
import numpy.typing as npt


def iou(
    pc_points: npt.NDArray[np.float64],
    pred_indices: npt.NDArray[np.float64],
    gt_indices: npt.NDArray[np.float64],
) -> np.float64:
    """
    :param pc_points: source point cloud
    :param pred_indices: indices of points that belong to one plane obtained as a result of segmentation
    :param gt_indices: indices of points belonging to the reference plane
    :return: iou metric value for plane
    """
    intersection = np.intersect1d(pred_indices, gt_indices)
    union = np.union1d(pred_indices, gt_indices)
    return intersection.size / union.size
