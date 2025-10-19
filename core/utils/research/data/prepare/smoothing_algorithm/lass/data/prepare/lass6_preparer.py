import numpy as np
import pandas as pd

from core.utils.research.data.prepare.smoothing_algorithm import IdentitySA
from core.utils.research.data.prepare.smoothing_algorithm.lass.data.prepare.lass3_preparer import Lass3Preparer


class Lass6Preparer(Lass3Preparer):

	def __init__(
			self,
			c_x: int,
			c_y: int,
			seq_size: int,
			*args,
			a: float = 1,
			f: float = 1,
			target_mean: float = 1.0,
			target_std: float = 0.3,
			noise: float = 0,
			seq_start: int = 0,
			**kwargs
	):
		super().__init__(
			*args,
			sa=IdentitySA(),
			shift=0,
			df=self.__generate_df(seq_size, seq_start),
			granularity=1,
			clean_df=False,
			**kwargs
		)
		self.__c_x = c_x
		self.__c_y = c_y
		self.__a = np.reshape(self._init_amplitudes(c_x, a), (1, -1, 1))
		self.__f = np.reshape(self._init_frequencies(c_x, f), (1, -1, 1)) * np.pi
		self.__noise = noise
		self.__tm, self.__ts = target_mean, target_std

	@staticmethod
	def __generate_df(size: int, start: int = 0) -> pd.DataFrame:
		df = pd.DataFrame(columns=["c"])
		df["c"] = np.arange(start, start + size)
		return df

	@staticmethod
	def _init_amplitudes(n: int, a: float) -> np.ndarray:
		return (1/(np.arange(n) + 1))**a

	@staticmethod
	def _init_frequencies(n: int, f: float) -> np.ndarray:
		return (0.1*(np.arange(n)+1))**f

	def __norm(self, x):
		return ((x) * self.__ts / np.std(x)) + self.__tm

	def _generate_shift(self, sequences: np.ndarray) -> np.ndarray:
		random = self._get_sequence_random(sequences)
		return random.random((sequences.shape[0], self.__c_x))*2*np.pi

	def _generate_noise(self, x: np.ndarray) -> np.ndarray:
		return self.__noise * 2 * (np.random.random((x.shape[0], x.shape[-1])) - 0.5)

	def _stack_noisy_and_smoothed(self, sequences: np.ndarray) -> np.ndarray:
		s = np.reshape(self._generate_shift(sequences), (-1, self.__c_x, 1))
		xs = self.__a * np.sin(
			s + self.__f*np.expand_dims(sequences/sequences.shape[1], axis=1)
		)

		x = self.__norm(np.sum(xs, axis=1) + self._generate_noise(sequences))
		y = self.__norm(np.sum(xs[:, :self.__c_y], axis=1))

		return np.stack([x, y], axis=1)

