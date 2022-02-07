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
        point_cloud: np.ndarray,
        points_predicted_indices: np.ndarray,
        points_gt_indices: np.ndarray,
    ) -> np.float64:
        """
        :param point_cloud: source point cloud
        :param points_predicted_indices: indices of point obtained as a result of segmentation
        :param points_gt_indices: indices of points belonging to the reference plane
        :return:
        """
        intersection = np.intersect1d(points_predicted_indices, points_gt_indices)
        union = np.union1d(points_predicted_indices, points_gt_indices)
        return intersection.size / union.size

    def calculate_metric_mean(
        self,
        point_cloud: np.ndarray,
        point_cloud_predicted_labels: np.ndarray,
        point_cloud_gt_labels: np.ndarray,
    ) -> np.float64:
        """
        :param point_cloud: source point cloud
        :param point_cloud_predicted_labels: segmented point cloud labels
        :param point_cloud_gt_labels: point cloud reference labels
        :return:
        """
        plane_predicted_dict = group_indices_by_labels(point_cloud_predicted_labels)
        plane_gt_dic = group_indices_by_labels(point_cloud_gt_labels)
        unique = np.unique(point_cloud_predicted_labels)
        iou_mean_array = np.empty((1, 0), np.float64)

        for key in unique:
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
