from typing import Any, Dict

import numpy as np
from nptyping import NDArray

import src.metrics.metrics
from src.utils.metrics_utils import __group_indices_by_labels, __are_nearly_overlapped


def __multi_value_benchmark(
    pc_points: NDArray[(Any, 3), np.float64],
    pred_labels: NDArray[Any, np.int32],
    gt_labels: NDArray[Any, np.int32],
    overlap_threshold: np.float64 = 0.8,
) -> Dict[str, np.float64]:
    correctly_segmented_amount = 0
    plane_predicted_dict = __group_indices_by_labels(pred_labels)
    plane_gt_dict = __group_indices_by_labels(gt_labels)
    if src.metrics.metrics.UNSEGMENTED_LABEL in plane_predicted_dict:
        del plane_predicted_dict[src.metrics.metrics.UNSEGMENTED_LABEL]
    if src.metrics.metrics.UNSEGMENTED_LABEL in plane_gt_dict:
        del plane_gt_dict[src.metrics.metrics.UNSEGMENTED_LABEL]
    predicted_amount = len(plane_predicted_dict)
    gt_amount = len(plane_gt_dict)
    under_segmented_amount = 0
    noise_amount = 0

    overlapped_predicted_by_gt = {label: [] for label in plane_gt_dict.keys()}

    for predicted_label, predicted_plane in plane_predicted_dict.items():
        overlapped_gt_planes = []
        for gt_label, gt_plane in plane_gt_dict.items():
            are_well_overlapped = __are_nearly_overlapped(
                predicted_plane, gt_plane, overlap_threshold
            )
            if are_well_overlapped:
                overlapped_gt_planes.append(gt_plane)
                overlapped_predicted_by_gt[gt_label].append(predicted_label)

        if len(overlapped_gt_planes) > 0:
            correctly_segmented_amount += 1
        else:
            noise_amount += 1

        if len(overlapped_gt_planes) > 1:
            under_segmented_amount += 1

    over_segmented_amount = 0
    missed_amount = 0
    for overlapped in overlapped_predicted_by_gt.values():
        if len(overlapped) > 1:
            over_segmented_amount += 1
        elif len(overlapped) == 0:
            missed_amount += 1

    return {
        "precision": correctly_segmented_amount / predicted_amount,
        "recall": correctly_segmented_amount / gt_amount,
        "under_segmented": under_segmented_amount / predicted_amount,
        "over_segmented": over_segmented_amount / gt_amount,
        "missed": missed_amount / gt_amount,
        "noise": noise_amount / predicted_amount,
    }
