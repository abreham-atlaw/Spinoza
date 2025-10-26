from abc import ABC, abstractmethod

import numpy as np


class Transformation(ABC):

	@abstractmethod
	def _transform(self, x: np.ndarray) -> np.ndarray:
		pass

	def transform(self, x: np.ndarray) -> np.ndarray:
		return self._transform(x.copy())

	def __call__(self, *args, **kwargs):
		return self.transform(*args, **kwargs)

	def __str__(self):
		return f"{self.__class__.__name__}"