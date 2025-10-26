import numpy as np

from .transformation import Transformation


class VerticalShiftTransformation(Transformation):

	def __init__(self, shift: float = 0.1):
		self.__shift = shift

	def _transform(self, x: np.ndarray) -> np.ndarray:
		return x + ((np.random.random((x.shape[0],) + (1,)*(x.ndim-1)) - 0.5) * np.mean(x, axis=tuple(range(1, x.ndim)), keepdims=True) * self.__shift)
