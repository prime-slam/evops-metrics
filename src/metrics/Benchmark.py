from abc import abstractmethod, ABC

import numpy as np


class Benchmark(ABC):
    @property
    def metric_name(self):
        return self.get_metric_name()

    @abstractmethod
    def get_metric_name(self) -> str:
        pass

    @abstractmethod
    def calculate_metric(
        self,
        point_cloud: np.ndarray,
        points_predicted_indices: np.ndarray,
        points_gt_indices: np.ndarray,
    ) -> np.float64:
        """
        :param point_cloud: source point cloud
        :param points_predicted_indices: indices of points that belong to one plane obtained as a result of segmentation
        :param points_gt_indices: indices of points belonging to the reference plane
        :return:
        """
        pass

    @abstractmethod
    def calculate_cumulative_metric(
        self,
        point_cloud: np.ndarray,
        point_cloud_predicted_labels: np.ndarray,
        point_cloud_gt_labels: np.ndarray,
    ) -> np.float64:
        """
        :param point_cloud: source point cloud
        :param point_cloud_predicted_labels: labels of points obtained as a result of segmentation
        :param point_cloud_gt_labels: reference labels of point cloud
        :return:
        """
        pass
