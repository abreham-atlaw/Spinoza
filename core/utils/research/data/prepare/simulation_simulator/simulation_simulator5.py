import typing

import numpy as np

from lib.utils.logger import Logger
from .simulation_simulator4 import SimulationSimulator4


class SimulationSimulator5(SimulationSimulator4):

	def __init__(
			self,
			*args,
			anchored_channels: typing.Tuple[str] = ('c', 'o', 'l', 'h'),
			anchor_channel: str = 'c',
			log_returns: bool = True,
			anchor_map: typing.Dict[str, str] = None,
			**kwargs,
	):
		super().__init__(*args, **kwargs)
		self.__anchored_channels = anchored_channels
		self.__anchor_channel = anchor_channel
		self.__log_returns = log_returns
		self.__anchor_map = anchor_map if anchor_map is not None else {
			channel: anchor_channel
			for channel in anchored_channels
		}
		Logger.info(
			f"Initializing {self.__class__.__name__} with anchor_map: {self.__anchor_map}"
			f" and log_returns: {log_returns} "
		)

	def __anchor_returns(self, sequences: np.ndarray) -> np.ndarray:
		y = sequences[..., 1:].copy()

		for anchored, anchor in self.__anchor_map.items():
			anchored_idx, anchor_idx = self._x_columns.index(anchored), self._x_columns.index(anchor)
			y[:, anchored_idx] /= (sequences[:, anchor_idx, :-1] + 1e-9)

		return y

	def _prepare_y_stack(self, sequences: np.ndarray) -> np.ndarray:
		sequences = self.__anchor_returns(sequences)
		if self.__log_returns:
			sequences = np.log(sequences)
		return sequences

	def _prepare_returns(self, sequences: np.ndarray) -> np.ndarray:
		return sequences[:, -1]
