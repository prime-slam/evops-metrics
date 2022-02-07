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
        pc_points: np.ndarray,
        pred_indices: np.ndarray,
        gt_indices: np.ndarray,
    ) -> np.float64:
        """
        :param pc_points: source point cloud
        :param pred_indices: indices of points that belong to one plane obtained as a result of segmentation
        :param gt_indices: indices of points belonging to the reference plane
        :return:
        """
        pass

    @abstractmethod
    def calculate_metric_mean(
        self,
        pc_points: np.ndarray,
        pred_labels: np.ndarray,
        gt_labels: np.ndarray,
    ) -> np.float64:
        """
        :param pc_points: source point cloud
        :param pred_labels: labels of points obtained as a result of segmentation
        :param gt_labels: reference labels of point cloud
        :return:
        """
        pass
