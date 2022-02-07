import numpy as np

from src.metrics.Benchmark import Benchmark
from src.utils.metrics import (
    get_dictionary_indices_of_current_label,
    planes_intersection_indices,
    planes_union_indices,
)


class IoUBenchmark(Benchmark):
    def get_metric_name(self) -> str:
        return "IoU"

    def calculate_metric(
        self, point_cloud: np.ndarray, plane_predicted: np.ndarray, plane_gt: np.ndarray
    ) -> np.float64:
        intersection = planes_intersection_indices(plane_predicted, plane_gt)
        union = planes_union_indices(plane_predicted, plane_gt)
        return intersection.size / union.size

    def calculate_cumulative_metric(
        self,
        point_cloud: np.ndarray,
        point_cloud_predicted: np.ndarray,
        point_cloud_gt: np.ndarray,
    ) -> np.float64:
        plane_predicted_dict = get_dictionary_indices_of_current_label(
            point_cloud_predicted
        )
        plane_gt_dic = get_dictionary_indices_of_current_label(point_cloud_gt)

        iou_mean_array = np.empty((1, 0), np.float64)
        for key, indices in enumerate(point_cloud_predicted):
            if key in plane_gt_dic:
                iou_mean_array = np.append(
                    iou_mean_array,
                    self.calculate_metric(
                        point_cloud, plane_predicted_dict[key], plane_gt_dic[key]
                    ),
                )

        if iou_mean_array.size == 0:
            return 0
        return iou_mean_array.mean()
