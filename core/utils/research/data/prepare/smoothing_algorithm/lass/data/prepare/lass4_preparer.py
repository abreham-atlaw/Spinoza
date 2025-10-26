import numpy as np

from .lass3_preparer import Lass3Preparer


class Lass4Preparer(Lass3Preparer):

	def _stack_noisy_and_smoothed(self, sequences: np.ndarray) -> np.ndarray:
		x = super()._stack_noisy_and_smoothed(sequences)
		x[:, 1, :] += ((x[:, 0, :1] - x[:, 1, :1])*np.random.random((x.shape[0], 1)))
		return x

	def _prepare_x(self, sequences: np.ndarray) -> np.ndarray:
		x = super()._prepare_x(sequences)
		x = x[np.any(x[:, 1] != 0, axis=1)]
		return x

	@staticmethod
	def __delta(x: np.ndarray):
		return x[:, 1:] - x[:, :-1]

	def _prepare_y(self, sequences: np.ndarray) -> np.ndarray:
		x = self._stack_noisy_and_smoothed(sequences)[:, 1, :]
		y = self.__delta(x)
		y = y.reshape((-1, 1))
		return y
