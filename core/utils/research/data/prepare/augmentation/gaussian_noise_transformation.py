import numpy as np

from .transformation import Transformation


class GaussianNoiseTransformation(Transformation):

	def __init__(self, mean: float = 0.0, r_std: float = 0.01, a_std: float = None):
		self.__mean = mean
		self.__r_std = r_std
		self.__a_std = a_std

	def _transform(self, x: np.ndarray) -> np.ndarray:
		std = self.__a_std if self.__a_std is not None else self.__r_std * np.std(x)
		return x + np.random.normal(self.__mean, std, x.shape)
