import random
import typing

import numpy as np

from core.agent.action import TraderAction, Action
from core.environment.trade_state import TradeState
from lib.rl.agent.utils.state_predictor import StatePredictor
from lib.rl.environment import ModelBasedState
from lib.utils.cache.decorators import CacheDecorators


class LegacyStatePredictor(StatePredictor):

	def __init__(self, *args, extra_len: int, **kwargs):
		super().__init__(*args, **kwargs)
		self.__extra_len = extra_len

	@staticmethod
	def __get_target_instrument(state, action, final_state) -> typing.Tuple[str, str]:
		if isinstance(action, TraderAction):
			return action.base_currency, action.quote_currency
		if final_state is None:
			return random.choice(state.get_market_state().get_tradable_pairs())
		for base_currency, quote_currency in final_state.get_market_state().get_tradable_pairs():
			if not np.all(final_state.get_market_state().get_state_of(base_currency,
																	  quote_currency) == state.get_market_state().get_state_of(
					base_currency, quote_currency)):
				return base_currency, quote_currency
		return final_state.get_market_state().get_tradable_pairs()[0]

	def __prepare_model_input(
			self,
			states: typing.List[TradeState],
			actions: typing.List[typing.Optional[Action]],
			target_instruments: typing.List[typing.Tuple[str, str]]
	) -> np.ndarray:
		x = np.stack([
			state.get_market_state().get_state_of(base_currency, quote_currency)
			for state, (base_currency, quote_currency) in zip(states, target_instruments)
		], axis=0)

		x = np.concatenate([
			x,
			np.zeros((x.shape[0], self.__extra_len))
		], axis=1)

		return x

	@CacheDecorators.cached_method()
	def prepare_input(
			self,
			states: typing.List[ModelBasedState],
			actions: typing.List[typing.Any],
			*args, **kwargs
	) -> np.ndarray:
		final_states = kwargs.get("final_states", [None]*len(states))

		target_instruments = [
			self.__get_target_instrument(state, action, final_state)
			for state, action, final_state in zip(states, actions, final_states)
		]
		return self.__prepare_model_input(states, actions, target_instruments)
