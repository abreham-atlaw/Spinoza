import typing

import numpy as np

from core.environment.trade_state import TradeState
from core.utils.research.data.prepare.utils.data_prep_utils import DataPrepUtils
from .training_target_builder import TrainingTargetBuilder
from core.agent.action import Action, TraderAction, ActionSequence


class LegacyTargetBuilder(TrainingTargetBuilder):

	def __init__(self, bounds: typing.List[float]):
		super().__init__()
		self.__bounds = bounds

	@staticmethod
	def __get_target_instrument(state: TradeState, action: Action, final_state: TradeState) -> typing.Tuple[str, str]:
		if isinstance(action, TraderAction):
			return action.base_currency, action.quote_currency
		if isinstance(action, ActionSequence):
			return LegacyTargetBuilder.__get_target_instrument(state, action.actions[-1], final_state)
		for base_currency, quote_currency in final_state.get_market_state().get_tradable_pairs():
			if not np.all(
					final_state.get_market_state().get_state_of(base_currency, quote_currency)
					== state.get_market_state().get_state_of(base_currency, quote_currency)):
				return base_currency, quote_currency
		return final_state.get_market_state().get_tradable_pairs()[0]

	def build_target(self, state: TradeState, action: Action, final_state: TradeState, value: float) -> np.ndarray:
		instrument = self.__get_target_instrument(state, action, final_state)
		percentage = final_state.get_market_state().get_current_price(*instrument) / state.get_market_state().get_current_price(*instrument)
		bound_idx = DataPrepUtils.find_bound_index(self.__bounds, percentage)
		output = np.zeros(len(self.__bounds) + 2)
		output[bound_idx] = 1
		output[-1] = value / state.get_agent_state().get_balance()
		return output