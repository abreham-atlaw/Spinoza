import numpy as np

from core.utils.research.data.prepare.smoothing_algorithm import SmoothingAlgorithm

from .lass_preparer import LassPreparer


class LassPreparer2(LassPreparer):

	def __init__(
			self,
			sa: SmoothingAlgorithm,
			shift: int,
			block_size: int,
			*args,
			**kwargs
	):
		super().__init__(sa, shift, block_size, *args, **kwargs)
		self.__initial_reduction = max(0, self._sa.reduction - self._shift)
		self.__smoothed_reduction = max(0, self._shift - self._sa.reduction)

	def _get_input_block_size(self, block_size: int) -> int:
		return block_size + max(self._shift, self._sa.reduction)

	def _stack_noisy_and_smoothed(self, sequences: np.ndarray) -> np.ndarray:
		noisy = sequences[:, max(0, self.__initial_reduction):-self._shift]
		smoothed = self._sa.apply_on_batch(sequences)[:, self.__smoothed_reduction:]
		return np.stack((noisy, smoothed), axis=1)

	def _prepare_x(self, sequences: np.ndarray) -> np.ndarray:
		data = self._stack_noisy_and_smoothed(sequences)
		data[:, 1, -1] = 0
		return data
