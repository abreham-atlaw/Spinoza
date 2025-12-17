import typing
from abc import ABC

import numpy as np

from lib.rl.environment import ModelBasedState
from lib.utils.cache import Cache
from lib.utils.logger import Logger
from .drmca import TraderDeepReinforcementMonteCarloAgent
from core import Config
from core.environment.trade_state import TradeState


class DirectProbabilityDistributionAgent(TraderDeepReinforcementMonteCarloAgent, ABC):

	def __init__(
			self,
			*args,
			use_direct_distribution: bool = Config.AGENT_USE_DIRECT_DISTRIBUTION,
			importance_threshold: float = Config.AGENT_POSSIBLE_STATES_IMPORTANCE_THRESHOLD,
			**kwargs,
	):
		super().__init__(*args, **kwargs)

		self.__probability_store = Cache(key_func=lambda state: id(state))
		self.__importance_threshold = importance_threshold
		self.__enabled = use_direct_distribution
		Logger.info(f"Direct probability distribution enabled: {self.__enabled}")

	def __get_transition_probability_distribution(
			self,
			state: TradeState,
			base_currency: str,
			quote_currency: str,
			action: typing.Any
	) -> np.ndarray:
		x = np.expand_dims(
			self._prepare_model_input(state, action, (base_currency, quote_currency)),
			axis=0
		)
		predictions, _ = self._parse_model_output(self._predict(self._transition_model, x)[0])
		return predictions

	def __trim_possible_values(
			self,
			values: np.ndarray,
			probabilities: np.ndarray
	) -> typing.Tuple[np.ndarray, np.ndarray]:

		probabilities = probabilities / np.sum(probabilities, axis=-1, keepdims=True)
		importance = probabilities / np.max(probabilities, axis=-1, keepdims=True)

		mask = importance > self.__importance_threshold

		return values[:, mask], probabilities[mask]

	def __get_possible_values(
			self,
			state: TradeState,
			action: typing.Any,
			base_currency: str,
			quote_currency: str
	) -> typing.Tuple[np.ndarray, np.ndarray]:
		values = self._get_possible_channel_values(state, base_currency, quote_currency)

		probabilities = self.__get_transition_probability_distribution(state, base_currency, quote_currency, action)
		probabilities = np.reshape(probabilities, (-1, probabilities.shape[-1]))

		probabilities = np.product(self._enumerate_channel_combinations(probabilities), axis=0)

		values, probabilities = self.__trim_possible_values(values, probabilities)

		return values, probabilities

	def _simulate_instrument_change_bound_mode(
			self,
			state: TradeState,
			base_currency: str,
			quote_currency: str,
			action: typing.Any
	) -> typing.List[TradeState]:

		if not self.__enabled:
			return super()._simulate_instrument_change_bound_mode(state, base_currency, quote_currency, action)

		states = []

		possible_values, probabilities = self.__get_possible_values(state, action, base_currency, quote_currency)

		for i in range(possible_values.shape[1]):
			new_state = state.__deepcopy__()
			new_state.recent_balance = state.get_agent_state().get_balance()

			new_value = np.expand_dims(possible_values[:, i], axis=1)

			new_state.get_market_state().update_state_of(
				base_currency,
				quote_currency,
				new_value
			)
			self._simulate_trades_triggers(new_state, (base_currency, quote_currency))

			self.__probability_store.store(new_state, probabilities[i])

			states.append(new_state)

		return states

	def _get_expected_transition_probability_distribution(
			self, initial_states: typing.List[ModelBasedState], action: typing.List[typing.Any], final_states: typing.List[ModelBasedState]
	) -> typing.List[float]:

		if not self.__enabled:
			return super()._get_expected_transition_probability_distribution(initial_states, action, final_states)

		values = [
			float(self.__probability_store.retrieve(final_states[i]))
			for i in range(len(final_states))
		]
		assert None not in values
		return values
