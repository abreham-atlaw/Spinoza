import typing
from datetime import datetime
from typing import *
from abc import ABC

import tensorflow as tf
import numpy as np

from concurrent.futures import ThreadPoolExecutor, as_completed


from core import Config
from core.agent.utils.cache import Cache
from core.utils.research.data.prepare.utils.data_prep_utils import DataPrepUtils
from core.utils.research.model.model.utils import WrappedModel
from lib.rl.agent import DNNTransitionAgent
from lib.rl.agent.dta import TorchModel
from lib.utils.logger import Logger
from core.environment.trade_state import TradeState, AgentState
from core.environment.trade_environment import TradeEnvironment
from core.agent.trader_action import TraderAction
from core.agent.utils.dnn_models import KerasModelHandler
from lib.utils.math import softmax
from temp import stats

from lib.utils.torch_utils.model_handler import ModelHandler


class TraderDNNTransitionAgent(DNNTransitionAgent, ABC):

	def __init__(
			self,
			*args,
			state_change_delta_model_mode=Config.AGENT_STATE_CHANGE_DELTA_MODEL_MODE,
			state_change_delta_bounds=Config.AGENT_STATE_CHANGE_DELTA_STATIC_BOUND,
			state_change_delta_bounds_epsilon=Config.AGENT_STATE_CHANGE_DELTA_STATIC_BOUND_EPSILON,
			update_agent=Config.UPDATE_AGENT,
			depth_mode=Config.AGENT_DEPTH_MODE,
			discount_function=Config.AGENT_DISCOUNT_FUNCTION,
			core_model=None,
			delta_model=None,
			use_softmax=Config.AGENT_USE_SOFTMAX,
			**kwargs
	):
		super().__init__(
			*args,
			depth=Config.AGENT_DEPTH,
			explore_exploit_tradeoff=Config.AGENT_EXPLOIT_EXPLORE_TRADEOFF,
			update_agent=update_agent,
			**kwargs
		)
		self.__state_change_delta_model_mode = state_change_delta_model_mode
		if isinstance(state_change_delta_bounds, list):
			state_change_delta_bounds = np.array(state_change_delta_bounds, dtype=np.float32)
		self._state_change_delta_bounds = state_change_delta_bounds
		self._simulation_state_change_delta_bounds = DataPrepUtils.apply_bound_epsilon(self._state_change_delta_bounds)
		self.__depth_mode = depth_mode
		self.environment: TradeEnvironment

		if core_model is None:
			Logger.info("Loading Core Model")
			core_model = TorchModel(
				WrappedModel(
					ModelHandler.load(Config.CORE_MODEL_CONFIG.path),
					seq_len=Config.MARKET_STATE_MEMORY,
					window_size=Config.AGENT_MA_WINDOW_SIZE
				)
			)
		self.set_transition_model(core_model)

		self.__delta_model = None
		if state_change_delta_model_mode:
			self.__delta_model = delta_model
			if delta_model is None:
				Logger.info("Loading Delta Model")
				self.__delta_model = KerasModelHandler.load_model(Config.DELTA_MODEL_CONFIG.path)

		self.__state_change_delta_cache = {}

		Logger.info(f"Using discount function: {discount_function}")
		self.__discount_function = discount_function

		self.__use_softmax = use_softmax
		self.__dta_output_cache = Cache()

	def _find_gap_index(self, number: float) -> int:
		boundaries = self._state_change_delta_bounds
		for i in range(len(boundaries)):
			if number < boundaries[i]:
				return i
		return len(boundaries)

	def __check_and_add_depth(self, input_: np.ndarray, depth: int) -> np.ndarray:
		if self.__depth_mode:
			input_ = np.append(input_, depth)
		return input_

	@staticmethod
	def _get_target_instrument(state, action, final_state) -> typing.Tuple[str, str]:
		if action is not None:
			return action.base_currency, action.quote_currency
		for base_currency, quote_currency in final_state.get_market_state().get_tradable_pairs():
			if not np.all(final_state.get_market_state().get_state_of(base_currency, quote_currency) == state.get_market_state().get_state_of(base_currency, quote_currency)):
				return base_currency, quote_currency
		return final_state.get_market_state().get_tradable_pairs()[0]
		# raise ValueError("Initial State and Final state are the same.")  # TODO: FIND ANOTHER WAY TO HANDLE THIS.

	def _prepare_single_dta_input(self, state: TradeState, action: TraderAction, final_state: TradeState) -> np.ndarray:

		base_currency, quote_currency = self._get_target_instrument(state, action, final_state)
		return self.__check_and_add_depth(
			state.get_market_state().get_state_of(base_currency, quote_currency),
			state.get_depth()
		).astype(np.float32)

	def _prepare_dta_input(self, states: typing.List[TradeState], actions: typing.List[TraderAction], final_states: typing.List[TradeState]) -> np.ndarray:
		return np.array([
			self._prepare_single_dta_input(state, action, final_state)
			for state, action, final_state in zip(states, actions, final_states)
		])

	def _get_discount_factor(self, depth) -> float:
		if self.__discount_function is None:
			return super()._get_discount_factor(depth)
		return self.__discount_function(depth)

	def __get_state_change_delta(self, sequence: np.ndarray, direction: int, depth: Optional[int] = None) -> float:

		if direction == -1:
			direction = 0

		model_input = np.append(sequence, direction)
		if depth is not None:
			model_input = self.__check_and_add_depth(model_input, depth)
		model_input = model_input.reshape((1, -1))

		cache_key = model_input.tobytes()

		cached = self.__state_change_delta_cache.get(cache_key)
		if cached is not None:
			return cached

		if self.__state_change_delta_model_mode:
			return_value = self.__delta_model.predict(
				model_input
			).flatten()[0]

		else:
			if isinstance(self._state_change_delta_bounds, float):
				percentage = self._state_change_delta_bounds
			else:
				percentage = np.random.uniform(self._state_change_delta_bounds[0], self._state_change_delta_bounds[1])

			return_value = sequence[-1] * percentage

		self.__state_change_delta_cache[cache_key] = return_value

		return return_value

	def _single_prediction_to_transition_probability_bound_mode(
			self,
			initial_state: TradeState,
			output: np.ndarray,
			final_state: TradeState
	) -> float:

		def compute(initial_state: TradeState,
			output: np.ndarray,
			final_state: TradeState) -> float:

			probabilities = output.flatten()

			for base_currency, quote_currency in final_state.get_market_state().get_tradable_pairs():

				if np.all(
						final_state.get_market_state().get_state_of(
							base_currency,
							quote_currency
						) == initial_state.get_market_state().get_state_of(
							base_currency,
							quote_currency)
				):
					continue

				percentage = final_state.get_market_state().get_current_price(
					base_currency,
					quote_currency
				) / initial_state.get_market_state().get_current_price(
					base_currency,
					quote_currency
				)
				if self.__use_softmax:
					probabilities = softmax(probabilities)

				idx = self._find_gap_index(percentage)

				return float(probabilities[idx])
		return self.__dta_output_cache.cached_or_execute((initial_state, output.tobytes(), final_state), lambda: compute(initial_state, output, final_state))

	def __prediction_to_transition_probability_bound_mode(
			self,
			initial_states: typing.List[TradeState],
			outputs: np.ndarray,
			final_states: typing.List[TradeState]
	) -> typing.List[float]:
		return [
			self._single_prediction_to_transition_probability_bound_mode(
				initial_state, output, final_state
			)
			for initial_state, output, final_state in zip(initial_states, outputs, final_states)
		]

	def _prepare_dta_output(
			self,
			initial_states: typing.List[TradeState],
			output: np.ndarray,
			final_states: typing.List[TradeState]
	) -> typing.List[float]:

		if not self.__state_change_delta_model_mode:
			return self.__prediction_to_transition_probability_bound_mode(initial_states, output, final_states)

		predicted_value: float = float(tf.reshape(output, (-1,))[0])
		for base_currency, quote_currency in final_states.get_market_state().get_tradable_pairs():

			if np.all(final_states.get_market_state().get_state_of(base_currency, quote_currency) == initial_states.get_market_state().get_state_of(base_currency, quote_currency)):
				continue

			if final_states.get_market_state().get_current_price(base_currency, quote_currency) > initial_states.get_market_state().get_current_price(base_currency, quote_currency):
				return predicted_value

			return 1-predicted_value

	def _prepare_single_dta_train_output(self, initial_state: TradeState, action, final_state: TradeState) -> np.ndarray:
		for base_currency, quote_currency in final_state.get_market_state().get_tradable_pairs():

			if final_state.get_market_state().get_current_price(base_currency, quote_currency) == initial_state.get_market_state().get_current_price(base_currency, quote_currency):
				continue

			if final_state.get_market_state().get_current_price(base_currency, quote_currency) > initial_state.get_market_state().get_current_price(base_currency, quote_currency):
				return np.array([1])

			return np.array([0])

		return np.array([0.5])

	def _prepare_dta_train_output(self, initial_state: TradeState, action, final_state: TradeState) -> np.ndarray:
		return np.stack([self._prepare_single_dta_train_output(initial_state, action, final_state)])

	def _get_expected_instant_reward(self, state) -> float:
		return self._get_environment().get_reward(state)

	def __get_involved_instruments(self, open_trades: List[AgentState.OpenTrade]) -> List[Tuple[str, str]]:
		return list(set(
			[
				(open_trade.get_trade().base_currency, open_trade.get_trade().quote_currency)
				for open_trade in open_trades
			]
		))

	def _get_possible_states(self, state: TradeState, action: TraderAction) -> List[TradeState]:

		states = []

		if len(state.get_agent_state().get_open_trades()) != 0:
			states += self.__simulate_instruments_change(
				state,
				self.__get_involved_instruments(state.get_agent_state().get_open_trades())
			)

		elif action is None or action.action == TraderAction.Action.CLOSE:
			states += self.__simulate_instruments_change(
				state,
				state.get_market_state().get_tradable_pairs()
			)

		else:
			states += self.__simulate_instrument_change(state, action.base_currency, action.quote_currency)

		# states = [self.__simulate_action(mid_state, action) for mid_state in states]
		for mid_state in states:
			self.__simulate_action(mid_state, action)

		return states

	def __simulate_instruments_change(self, mid_state, instruments: List[Tuple[str, str]]) -> List[TradeState]:
		states = []
		for base_currency, quote_currency in instruments:
			states += self.__simulate_instrument_change(mid_state, base_currency, quote_currency)

		return states

	def __simulate_instrument_change_bound_mode(self, state: TradeState, base_currency: str, quote_currency: str) -> List[TradeState]:
		states = []

		original_value = state.get_market_state().get_state_of(base_currency, quote_currency)

		for j in range(len(self._simulation_state_change_delta_bounds)):
			new_state = state.__deepcopy__()
			new_state.recent_balance = state.get_agent_state().get_balance()
			new_value = np.array(original_value[-1] * self._simulation_state_change_delta_bounds[j], dtype=np.float32).reshape(1)
			new_state.get_market_state().update_state_of(
				base_currency,
				quote_currency,
				new_value
			)
			states.append(new_state)

		return states

	def __simulate_instrument_change(self, state: TradeState, base_currency: str, quote_currency: str) -> List[TradeState]:
		if not self.__state_change_delta_model_mode:
			return self.__simulate_instrument_change_bound_mode(state, base_currency, quote_currency)

		states = []

		original_value = state.get_market_state().get_state_of(base_currency, quote_currency)

		for j in [-1, 1]:
			new_state = state.__deepcopy__()
			new_state.recent_balance = state.get_agent_state().get_balance()
			new_state.get_market_state().update_state_of(
				base_currency,
				quote_currency,
				np.array(original_value[-1] + j*self.__get_state_change_delta(original_value, j, state.get_depth())).reshape(1)
			)
			states.append(new_state)

		return states

	def __simulate_action(self, state: TradeState, action: TraderAction):  # TODO: SETUP CACHER
		# state = copy.deepcopy(state)

		if action is None:
			return

		if action.action == TraderAction.Action.CLOSE:
			state.get_agent_state().close_trades(action.base_currency, action.quote_currency)
			return

		state.get_agent_state().open_trade(
			action,
			state.get_market_state().get_current_price(action.base_currency, action.quote_currency)
		)
