import numpy as np

from .lass_preparer_2 import LassPreparer2


class Lass5Preparer(LassPreparer2):

	def _prepare_x(self, sequences: np.ndarray) -> np.ndarray:
		x = self._stack_noisy_and_smoothed(sequences)
		return x[:, 0, :]

	def _prepare_y(self, sequences: np.ndarray) -> np.ndarray:
		y = self._stack_noisy_and_smoothed(sequences)
		return y[:, 1, :]
