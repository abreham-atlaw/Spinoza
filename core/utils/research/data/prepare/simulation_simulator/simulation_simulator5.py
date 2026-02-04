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
			**kwargs,
	):
		super().__init__(*args, **kwargs)
		self.__anchored_channels = anchored_channels
		self.__anchor_channel = anchor_channel
		self.__log_returns = log_returns
		Logger.info(f"Initializing {self.__class__.__name__} with anchored_channels: {anchored_channels}, anchor_channel: {anchor_channel} and log_returns: {log_returns}")

	def __anchor_returns(self, sequences: np.ndarray) -> np.ndarray:
		y = sequences[..., 1:].copy()
		anchored_channels_idxs = [i for i in range(y.shape[1]) if self._x_columns[i] in self.__anchored_channels]
		anchor_idx = self._x_columns.index(self.__anchor_channel)

		y[:, anchored_channels_idxs] /= (sequences[:, [anchor_idx], :-1] + 1e-9)
		return y

	def _prepare_y_stack(self, sequences: np.ndarray) -> np.ndarray:
		sequences = self.__anchor_returns(sequences)
		if self.__log_returns:
			sequences = np.log(sequences)
		return sequences

	def _prepare_returns(self, sequences: np.ndarray) -> np.ndarray:
		return sequences[:, -1]
