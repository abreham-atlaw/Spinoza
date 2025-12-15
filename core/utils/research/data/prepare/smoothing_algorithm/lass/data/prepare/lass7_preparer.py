import numpy as np

from core.utils.research.data.prepare.smoothing_algorithm import IdentitySA
from core.utils.research.data.prepare.utils.sinusoidal_decomposer import SinusoidalDecomposer
from .lass3_preparer import Lass3Preparer


class Lass7Preparer(Lass3Preparer):

	def __init__(
			self,
			decomposer: SinusoidalDecomposer,
			*args,
			**kwargs
	):
		super().__init__(
			*args,
			**kwargs,
			sa=IdentitySA(),
			shift=0
		)
		self.__decomposer = decomposer

	def _prepare_sequence(self, sequence: np.ndarray) -> np.ndarray:
		y = self.__decomposer.decompose(sequence)
		return np.stack([sequence, y], axis=0)

	def _stack_noisy_and_smoothed(self, sequences: np.ndarray) -> np.ndarray:
		return sequences
