import numpy as np

from .lass7_preparer import Lass7Preparer


class Lass8Preparer(Lass7Preparer):

	def __init__(self, *args, **kwargs):
		super().__init__(
			*args,
			**kwargs,
		)

	def _prepare_x(self, sequences: np.ndarray) -> np.ndarray:
		sequences = self._stack_noisy_and_smoothed(sequences)
		return sequences[:, 0]

	def _prepare_y(self, sequences: np.ndarray) -> np.ndarray:
		sequences = self._stack_noisy_and_smoothed(sequences)
		return sequences[:, 1]
