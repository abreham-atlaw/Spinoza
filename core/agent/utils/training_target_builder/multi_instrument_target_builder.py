import typing

import numpy as np

from core.environment.trade_state import TradeState
from core.utils.research.data.prepare.utils.data_prep_utils import DataPrepUtils
from .training_target_builder import TrainingTargetBuilder
from core.agent.action import Action


class MultiInstrumentTargetBuilder(TrainingTargetBuilder):

	def __init__(
			self,
			channels: typing.Tuple[int,...],
			bounds: typing.List[float],
			anchor_channel: str = 'c',
	):
		self.__anchor_channel_idx = channels.index(anchor_channel)
		self.__bounds = bounds

	def __build_instrument_tp_target(self, state: TradeState, final_state: TradeState, instrument: typing.Tuple[str, str]) -> np.ndarray:
		initial_values = state.get_market_state().get_channels_state(*instrument)[self.__anchor_channel_idx, -1]
		final_values = final_state.get_market_state().get_channels_state(*instrument)[:, -1]

		returns = np.log(final_values / (initial_values + 1e-9))
		classes = np.array([
			DataPrepUtils.find_bound_index(self.__bounds, r)
			for r in returns
		])

		encoded = DataPrepUtils.one_hot_encode(classes, len(self.__bounds) + 1)
		return encoded

	def __build_tp_target(self, state: TradeState, final_state: TradeState) -> np.ndarray:
		return np.concatenate([
			self.__build_instrument_tp_target(state, final_state, instrument)
			for instrument in state.get_market_state().get_tradable_pairs()
		], axis=0)

	@staticmethod
	def __merge_tp_value(tp_target: np.ndarray, value_target: float):
		y = np.concatenate([
			tp_target,
			np.zeros((tp_target.shape[0], 1))
		], axis=1)
		y[0, -1] = value_target
		return y

	def build(self, state: TradeState, action: Action, final_state: TradeState, value: float) -> np.ndarray:
		tp_target = self.__build_tp_target(state, final_state)
		value_target = value / state.get_agent_state().get_balance()
		y = self.__merge_tp_value(tp_target, value_target)
		return y
