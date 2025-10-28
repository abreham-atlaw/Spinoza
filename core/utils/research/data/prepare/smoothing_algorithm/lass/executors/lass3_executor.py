import typing

import numpy as np

from core.utils.research.model.model.savable import SpinozaModule
from lib.utils.logger import Logger
from .lass_executor import LassExecutor


class Lass3Executor(LassExecutor):

	def __init__(
			self,
			*args,
			padding: int = 0,
			left_align: bool = False,
			verbose_threshold: int = int(1e4),
			**kwargs
	):
		super().__init__(*args, **kwargs)
		self._padding = padding
		self.__target_size = None
		self._left_align = left_align
		self.__verbose_threshold = verbose_threshold

	def set_model(self, model: SpinozaModule):
		super().set_model(model)
		self.__target_size = self._window_size - 2 * self._padding

	@property
	def supports_batch_execution(self) -> bool:
		return True

	def _get_source_block(self, x: np.ndarray, target: typing.Tuple[int, int]) -> typing.Tuple[int, int]:
		if target[0] == 0:
			return 0, self._window_size
		if target[1] + self._padding >= x.shape[1]:
			return x.shape[1] - self._window_size, x.shape[1]

		return target[0] - self._padding, target[1] + self._padding

	def _get_next_target(self, x: np.ndarray, last_target: typing.Union[None, typing.Tuple[int, int]]) -> typing.Tuple[int, int]:
		if last_target is None:
			return 0, self._window_size - self._padding
		if last_target[-1] + self.__target_size >= x.shape[1]:
			return x.shape[1] - self.__target_size, x.shape[1]
		return last_target[-1], last_target[-1] + self.__target_size

	def __extract_target(self, y_block: np.ndarray, target: typing.Tuple[int, int], source: typing.Tuple[int, int]) -> np.ndarray:
		return y_block[:, target[0]-source[0]:target[1]-source[0]]

	def __construct_input(self, x: np.ndarray, y: np.ndarray, i: int) -> np.ndarray:
		inputs = np.zeros((x.shape[0], 2, self._window_size))
		inputs[:, 0] = x
		if not self._left_align:
			inputs[:, 1, inputs.shape[-1]-i:] = y[:, :i]
		else:
			inputs[:, 1, :i] = y[:, i]
		return inputs

	def _process_prediction(self, x: np.ndarray, y: np.ndarray) -> np.ndarray:
		return y.reshape((y.shape[0],))

	def __execute_block(self, x: np.ndarray, y: np.ndarray, start: int) -> np.ndarray:
		y = y.copy()
		for i in range(start, y.shape[1]):
			inputs = self.__construct_input(x, y, i)
			prediction = self._model.predict(inputs)
			y[:, i] = self._process_prediction(inputs, prediction)

			if self.__verbose_threshold is not None and x.shape[0] >= self.__verbose_threshold:
				Logger.info(f"Executed Block({x.shape[0]}, {start}-{y.shape[1]}) {(i+1)*100/x.shape[1] :.2f}%...", end="\r")

		return y

	def _init_y(self, X: np.ndarray) -> np.ndarray:
		return np.zeros_like(X)

	def _execute(self, X: np.ndarray) -> np.ndarray:

		is_flat = len(X.shape) == 1

		if is_flat:
			X = np.expand_dims(X, axis=0)

		target: typing.Union[None, typing.Tuple[int, int]] = None
		y = self._init_y(X)

		while target is None or target[-1] != X.shape[1]:
			target = self._get_next_target(X, target)
			source = self._get_source_block(X, target)
			y_block = self.__execute_block(
				X[:, source[0]:source[1]],
				y[:, source[0]:source[1]],
				target[0]-source[0]
			)
			y[:, target[0]:target[1]] = self.__extract_target(y_block, target, source)


		if is_flat:
			y = y.flatten()

		return y
