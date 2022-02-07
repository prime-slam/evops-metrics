from abc import abstractmethod

import numpy as np


class Benchmark:
    @property
    def metric_name(self):
        return self.get_metric_name()

    @abstractmethod
    def get_metric_name(self) -> str:
        pass

    @abstractmethod
    def calculate_metric(
        self, point_cloud: np.ndarray, plane_predicted: np.ndarray, plane_gt: np.ndarray
    ) -> np.float64:
        pass

    @abstractmethod
    def calculate_cumulative_metric(
        self,
        point_cloud: np.ndarray,
        point_cloud_predicted: np.ndarray,
        point_cloud_gt: np.ndarray,
    ) -> np.float64:
        pass
