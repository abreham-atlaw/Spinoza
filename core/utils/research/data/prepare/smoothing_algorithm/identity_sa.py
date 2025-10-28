import numpy as np

from .smoothing_algorithm import SmoothingAlgorithm


class IdentitySA(SmoothingAlgorithm):

	@property
	def reduction(self) -> int:
		return 0

	def apply(self, x: np.ndarray) -> np.ndarray:
		return x
