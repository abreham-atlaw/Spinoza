import typing

import numpy as np

from core.environment.trade_state import TradeState
from core.utils.research.data.prepare.utils.data_prep_utils import DataPrepUtils
from .state_transition_sampler import StateTransitionSampler


class BasicStateTransitionSampler(StateTransitionSampler):

	def __init__(
			self,
			bounds: typing.List[float],
			channels: typing.List[int],
			simulated_channels: typing.List[int],
	):
		self._bounds = DataPrepUtils.apply_bound_epsilon(bounds)
		self._channels = channels
		self.__simulated_channels = simulated_channels
		self.__channels_idx = [i for i in range(len(channels)) if channels[i] in simulated_channels]

	def _get_possible_values(self, original_values: np.ndarray) -> np.ndarray:
		possible_values = original_values[self.__channels_idx][:, -1:] * self._bounds
		return possible_values

	def sample_next_values(self, state: TradeState, instrument: typing.Tuple[str, str]) -> np.ndarray:
		original_values = state.get_market_state().get_channels_state(instrument[0], instrument[1])[self.__channels_idx]
		possible_values = self._get_possible_values(original_values)
		return possible_values
