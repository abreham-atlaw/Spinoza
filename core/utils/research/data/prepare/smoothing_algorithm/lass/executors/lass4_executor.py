import typing

import numpy as np

from lib.utils.logger import Logger
from .lass3_executor import Lass3Executor


class Lass4Executor(Lass3Executor):

	def __init__(self, *args, **kwargs):
		super().__init__(*args, **kwargs)
		if self._padding < 1:
			Logger.warning(f"A minimum of 1 padding is recommended for Lass4. Current is {self._padding}")

	def _init_y(self, X: np.ndarray) -> np.ndarray:
		y = super()._init_y(X)
		y[0] = X[0]
		return y

	def _process_prediction(self, x: np.ndarray, y: np.ndarray) -> np.ndarray:
		y = x[0, 1, -1] + y.flatten()[0]
		return y

	def _get_next_target(self, x: np.ndarray, last_target: typing.Union[None, typing.Tuple[int, int]]) -> typing.Tuple[int, int]:
		target = super()._get_next_target(x, last_target)
		if last_target is None:
			target = 1, target[1]
		return target

	def _get_source_block(self, x: np.ndarray, target: typing.Tuple[int, int]) -> typing.Tuple[int, int]:
		source = super()._get_source_block(x, target)
		if target[0] == 1:
			source = 0, source[1]
		return source
