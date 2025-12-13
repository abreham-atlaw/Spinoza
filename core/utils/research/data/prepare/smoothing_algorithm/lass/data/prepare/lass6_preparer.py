import typing

import numpy as np
import pandas as pd

from core.utils.research.data.prepare.smoothing_algorithm import IdentitySA
from core.utils.research.data.prepare.smoothing_algorithm.lass.data.prepare.lass3_preparer import Lass3Preparer
from lib.utils.cache import Cache
from lib.utils.cache.decorators import CacheDecorators
from lib.utils.logger import Logger


class Lass6Preparer(Lass3Preparer):

	def __init__(
			self,
			c_x: int,
			c_y: int,
			seq_size: int,
			block_size: int,
			*args,
			a: typing.Union[float, np.ndarray] = 1,
			f: typing.Union[float, np.ndarray] = 1,
			target_mean: float = 1.0,
			target_std: float = 0.3,
			noise: float = 0,
			noise_p: float = 0,
			seq_start: int = 0,
			lag: int = 0,
			**kwargs
	):
		super().__init__(
			*args,
			sa=IdentitySA(),
			shift=0,
			df=self.__generate_df(seq_size, seq_start),
			granularity=1,
			clean_df=False,
			block_size=block_size + lag,
			**kwargs
		)

		if not isinstance(a, np.ndarray):
			Logger.info(f"Initializing Amplitudes...")
			a = self._init_amplitudes(c_x, a)

		if not isinstance(f, np.ndarray):
			Logger.info(f"Initializing Frequencies...")
			f = self._init_frequencies(c_x, f)

		for arr, name in zip([a, f], ["amplitudes", "frequencies"]):
			if arr.shape[0] != c_x:
				raise ValueError(f"Size of {name}, {arr.shape[0]}, doesn't match c_x, {c_x}.")

		self.__c_x = c_x
		self.__c_y = c_y
		self.__a = np.reshape(a, (1, -1, 1))
		self.__f = np.reshape(f, (1, -1, 1)) * np.pi
		self.__noise = noise
		self.__noise_p = noise_p
		self.__tm, self.__ts = target_mean, target_std
		self.__stack_cache = Cache(cache_size=1, key_func=lambda x: x.tobytes())
		self.__lag = lag

		if self.__noise_p % 2 == 0:
			Logger.warning(f"Using even noise power, {noise_p}. It is recommended to use odd noise power.")

		Logger.info(
			f"Initialized Lass6Preparer using lag={lag}, c_x={c_x}, c_y={c_y}, noise={noise}, noise_p={noise_p}, "
			f"target_mean={target_mean}, target_std={target_std}"
		)

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
		return self.__noise * ((2*(np.random.random((x.shape[0], x.shape[-1])) - 0.5)) ** self.__noise_p)

	def _apply_transformations(self, x: np.ndarray) -> np.ndarray:
		return x

	def __apply_lag(self, x: np.ndarray) -> np.ndarray:
		if self.__lag == 0:
			return x
		return np.stack([
			x[:, 0, self.__lag:],
			x[:, 1, :-self.__lag]
		], axis=1)

	def _stack_noisy_and_smoothed(self, sequences: np.ndarray) -> np.ndarray:

		cached = self.__stack_cache.retrieve(sequences)
		if cached is not None:
			return cached
		s = np.reshape(self._generate_shift(sequences), (-1, self.__c_x, 1))
		xs = self.__a * np.sin(
			s + self.__f*np.expand_dims(sequences/sequences.shape[1], axis=1)
		)

		x = self.__norm(np.sum(xs, axis=1) + self._generate_noise(sequences))
		y = self.__norm(np.sum(xs[:, :self.__c_y], axis=1))

		stack = np.stack([x, y], axis=1)
		stack = super()._apply_transformations(stack)

		stack = self.__apply_lag(stack)

		self.__stack_cache[sequences] = stack

		return stack
