import numpy as np

from src.metrics.Benchmark import Benchmark
from src.utils.metrics import (
    group_indices_by_labels,
)


class IoUBenchmark(Benchmark):
    def get_metric_name(self) -> str:
        return "IoU"

    def calculate_metric(
        self,
        pc_points: np.ndarray,
        pred_indices: np.ndarray,
        gt_indices: np.ndarray,
    ) -> np.float64:
        intersection = np.intersect1d(pred_indices, gt_indices)
        union = np.union1d(pred_indices, gt_indices)
        return intersection.size / union.size

    def calculate_metric_mean(
        self,
        pc_points: np.ndarray,
        pred_labels: np.ndarray,
        gt_labels: np.ndarray,
    ) -> np.float64:
        plane_predicted_dict = group_indices_by_labels(pred_labels)
        plane_gt_dict = group_indices_by_labels(gt_labels)
        unique_labels = np.unique(pred_labels)
        iou_mean_array = np.empty((1, 0), np.float64)

        for label in unique_labels:
            if label in plane_gt_dict:
                iou_mean_array = np.append(
                    iou_mean_array,
                    self.calculate_metric(
                        pc_points, plane_predicted_dict[label], plane_gt_dict[label]
                    ),
                )

        if iou_mean_array.size == 0:
            return 0

        return iou_mean_array.mean()
