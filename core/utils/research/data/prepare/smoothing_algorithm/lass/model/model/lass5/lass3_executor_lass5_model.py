import typing
import torch

from core.utils.research.model.model.savable import SpinozaModule


class Lass3ExecutorLass5Model(SpinozaModule):


	def __init__(
		self,
		block_size: int,
		padding: int,
		model: SpinozaModule,
	):
		self.args = {
			"model": model,
			"padding": padding,
			"block_size": block_size
		}
		super().__init__(input_size=(None, block_size), auto_build=False)
		self.model = model
		self._window_size = model.input_size[-1]
		self._padding = padding
		self.__target_size = self._window_size - 2 * self._padding
		self.init()

	@staticmethod
	def _process_prediction(
			x: torch.Tensor,
			y: torch.Tensor
	) -> torch.Tensor:
		return y[:, 0]

	@staticmethod
	def _init_y(X: torch.Tensor) -> torch.Tensor:
		return torch.zeros(X.shape[0], X.shape[1], device=X.device, dtype=X.dtype)

	def _get_source_block(
		self, x: torch.Tensor, target: typing.Tuple[int, int]
	) -> typing.Tuple[int, int]:
		if target[0] == 0:
			return 0, self._window_size
		if target[1] + self._padding >= x.shape[-1]:
			return x.shape[-1] - self._window_size, x.shape[-1]
		return target[0] - self._padding, target[1] + self._padding

	def _get_next_target(
		self, x: torch.Tensor, last_target: typing.Optional[typing.Tuple[int, int]]
	) -> typing.Tuple[int, int]:
		T = x.shape[-1]
		if last_target is None:
			return 0, self._window_size - self._padding
		if last_target[-1] + self.__target_size >= T:
			return T - self.__target_size, T
		return last_target[-1], last_target[-1] + self.__target_size

	@staticmethod
	def __extract_target(
			y_block: torch.Tensor,
		target: typing.Tuple[int, int],
		source: typing.Tuple[int, int],
	) -> torch.Tensor:
		return y_block[:, target[0] - source[0]: target[1] - source[0]]

	def __construct_input(
		self, x: torch.Tensor, y: torch.Tensor, i: int
	) -> torch.Tensor:
		B = x.shape[0]
		inputs = torch.zeros((B, 2, self._window_size), device=x.device, dtype=x.dtype)
		inputs[:, 0] = x
		inputs[:, 1, inputs.shape[-1] - i:] = y[:, :i]
		return inputs

	def __execute_block(
		self, x: torch.Tensor, y: torch.Tensor, start: int
	) -> torch.Tensor:
		y = y.clone()
		for i in range(start, y.shape[1]):
			inputs = self.__construct_input(x, y, i)
			prediction = self.model(inputs)
			values = self._process_prediction(inputs, prediction)
			y[:, i] = values
		return y

	def call(self, x: torch.Tensor) -> torch.Tensor:
		target: typing.Optional[typing.Tuple[int, int]] = None
		y = self._init_y(x)

		while target is None or target[-1] != x.shape[1]:
			target = self._get_next_target(x, target)
			source = self._get_source_block(x, target)

			y_block = self.__execute_block(
				x[:, source[0]: source[1]],
				y[:, source[0]: source[1]],
				target[0] - source[0],
			)
			y[:, target[0]: target[1]] = self.__extract_target(y_block, target, source)

		return y

	def export_config(self) -> typing.Dict[str, typing.Any]:
		return self.args