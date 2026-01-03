import numpy as np

from core.utils.research.data.prepare.smoothing_algorithm import IdentitySA
from core.utils.research.data.prepare.utils.sinusoidal_decomposer import SinusoidalDecomposer
from .lass3_preparer import Lass3Preparer


class Lass7Preparer(Lass3Preparer):

	def __init__(
			self,
			decomposer: SinusoidalDecomposer,
			*args,
			vertical_align: bool = False,
			**kwargs
	):
		super().__init__(
			*args,
			**kwargs,
			sa=IdentitySA(),
			shift=0
		)
		self.__decomposer = decomposer
		self.__vertical_align = vertical_align

	def _prepare_sequence(self, sequence: np.ndarray) -> np.ndarray:
		y = self.__decomposer.decompose(sequence)
		return np.stack([sequence, y], axis=0)

	def _prepare_sequence_stack(self, x: np.ndarray) -> np.ndarray:

		if self.__vertical_align:
			x[:, 1] += np.mean(x[:, 0], axis=-1, keepdims=True) - np.mean(x[:, 1], axis=-1,
																		  keepdims=True)
		return x

	def _stack_noisy_and_smoothed(self, sequences: np.ndarray) -> np.ndarray:
		return sequences
