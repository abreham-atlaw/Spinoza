import numpy as np

from .transformation import Transformation


class VerticalStretchTransformation(Transformation):

	def __init__(self, alpha: float):
		self.__alpha = alpha

	def _transform(self, x: np.ndarray) -> np.ndarray:
		anchor = np.expand_dims(
			np.random.random(x.shape[0]) * (np.max(x, axis=1) - np.min(x, axis=1)) + np.min(x, axis=1),
			axis=1
		)
		return (x - anchor)*self.__alpha + anchor


