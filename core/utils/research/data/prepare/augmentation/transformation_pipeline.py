import typing

import numpy as np

from .transformation import Transformation


class TransformationPipeline(Transformation):

	def __init__(self, transformations: typing.List[Transformation]):
		super().__init__()
		self.__transformations = transformations

	def _transform(self, x: np.ndarray) -> np.ndarray:
		for transformation in self.__transformations:
			x = transformation(x)
		return x
