import typing
from typing import *
from abc import ABC

import tensorflow as tf
import numpy as np

from core import Config
from core.agent.utils.cache import Cache
from core.agent.utils.state_transition_sampler import StateTransitionSampler
from core.di import AgentUtilsProvider
from lib.rl.agent import DNNTransitionAgent
from lib.utils.logger import Logger
from core.environment.trade_state import TradeState, AgentState, InsufficientFundsException
from core.environment.trade_environment import TradeEnvironment
from core.agent.action import TraderAction, Action, ActionSequence
from core.agent.utils.dnn_models import KerasModelHandler
from lib.utils.math import softmax


class TraderDNNTransitionAgent(DNNTransitionAgent, ABC):

	def __init__(
			self,
			*args,
			state_change_delta_model_mode=Config.AGENT_STATE_CHANGE_DELTA_MODEL_MODE,
			state_change_delta_bounds=Config.AGENT_STATE_CHANGE_DELTA_STATIC_BOUND,
			update_agent=Config.UPDATE_AGENT,
			depth_mode=Config.AGENT_DEPTH_MODE,
			discount_function=Config.AGENT_DISCOUNT_FUNCTION,
			delta_model=None,
			use_softmax=Config.AGENT_USE_SOFTMAX,
			use_multi_channels=Config.MARKET_STATE_USE_MULTI_CHANNELS,
			market_state_channels: typing.Tuple[str, ...] = Config.MARKET_STATE_CHANNELS,
			simulated_channels: typing.Tuple[str, ...] = Config.MARKET_STATE_SIMULATED_CHANNELS,
			state_transition_sampler: StateTransitionSampler = None,
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
		self.__depth_mode = depth_mode
		self.environment: TradeEnvironment

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
		self._use_multi_channels = use_multi_channels
		self.__market_state_channels = market_state_channels
		self.__simulated_channels = simulated_channels
		self.__close_channel, self.__high_channel, self.__low_channel = self.__init_channel_idxs(simulated_channels)
		self.__channels_map = [self.__market_state_channels.index(channel) for channel in self.__simulated_channels]
		self.__state_transition_sampler = state_transition_sampler if state_transition_sampler is not None else AgentUtilsProvider.provide_state_transition_sampler()
		Logger.info(f"Initializing TraderDNNTransitionAgent with multi_channels={use_multi_channels}, market_state_channels={market_state_channels}, simulated_channels={simulated_channels}")

	@staticmethod
	def __init_channel_idxs(channels: typing.Tuple[str, ...]) -> typing.Tuple[int, int, int]:
		close_channel = channels.index("c")
		high_channel = channels.index("h") if "h" in channels else close_channel
		low_channel = channels.index("l") if "l" in channels else close_channel

		return close_channel, high_channel, low_channel

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
		if isinstance(action, TraderAction):
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

		def compute(
				initial_state: TradeState,
				output: np.ndarray,
				final_state: TradeState
		) -> float:

			probabilities = output.reshape((-1, output.shape[-1]))

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

				percentage = (final_state.get_market_state().get_channels_state(
					base_currency,
					quote_currency
				)[:, -1] / initial_state.get_market_state().get_channels_state(
					base_currency,
					quote_currency
				)[:, -1])[self.__channels_map]

				if self.__use_softmax:
					probabilities = np.array([softmax(p) for p in probabilities])

				idxs = [self._find_gap_index(percentage[i]) for i in range(percentage.shape[0])]

				return float(np.product(probabilities[np.arange(probabilities.shape[0]), idxs]))
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

	@staticmethod
	def __get_involved_instruments(open_trades: List[AgentState.OpenTrade]) -> List[Tuple[str, str]]:
		return list(set(
			[
				(open_trade.get_trade().base_currency, open_trade.get_trade().quote_currency)
				for open_trade in open_trades
			]
		))

	def __simulate_trade_trigger(self, state: TradeState, trade: AgentState.OpenTrade):
		if trade.get_trade().stop_loss is None and trade.get_trade().take_profit is None:
			return

		instrument = trade.get_trade().base_currency, trade.get_trade().quote_currency

		current_price = state.get_market_state().get_channels_state(instrument[0], instrument[1])[:, -1]
		previous_price = trade.get_enter_value()

		percentage = current_price / previous_price

		direction = -1 if trade.get_trade().action == TraderAction.Action.SELL else 1

		stop_loss_channel = self.__low_channel if direction == 1 else self.__high_channel
		take_profit_channel = self.__high_channel if direction == 1 else self.__low_channel

		close_price = None

		if trade.get_trade().stop_loss is not None and (direction * percentage[stop_loss_channel] <= direction * trade.get_trade().stop_loss):
			close_price = previous_price * trade.get_trade().stop_loss

		elif trade.get_trade().take_profit is not None and (direction * percentage[take_profit_channel] >= direction * trade.get_trade().take_profit):
			close_price = previous_price * trade.get_trade().take_profit

		if close_price is not None:
			state.get_agent_state().close_trades(instrument[0], instrument[1], close_price=close_price)  # TODO: CLOSE SINGLE TRADE

	def _simulate_trades_triggers(self, state: TradeState, instrument: Tuple[str, str]):
		for trade in state.get_agent_state().get_open_trades(instrument[0], instrument[1]):
			self.__simulate_trade_trigger(state, trade)

	def _get_possible_states(self, state: TradeState, action: Action) -> List[TradeState]:

		involved_instruments = []

		if len(state.get_agent_state().get_open_trades()) != 0:
			involved_instruments += self.__get_involved_instruments(state.get_agent_state().get_open_trades())

		if isinstance(action, TraderAction):
			involved_instruments.append((action.base_currency, action.quote_currency))

		elif isinstance(action, ActionSequence):
			involved_instruments.extend([(action.base_currency, action.quote_currency) for action in action.actions])

		elif action is None and len(state.get_agent_state().get_open_trades()) == 0:
			involved_instruments = state.get_market_state().get_tradable_pairs()

		involved_instruments = list(set(involved_instruments))

		states = self.__simulate_instruments_change(state, involved_instruments, action)

		for mid_state in states:
			self._simulate_action(mid_state, action)

		return states

	def __simulate_instruments_change(self, mid_state, instruments: List[Tuple[str, str]], action) -> List[TradeState]:
		states = []
		for base_currency, quote_currency in instruments:
			ins_state = self.__simulate_instrument_change(mid_state, base_currency, quote_currency, action)
			for state in ins_state:
				state.simulated_instrument = (base_currency, quote_currency)
			states += ins_state

		return states

	@staticmethod
	def _enumerate_channel_combinations(possible_values: np.ndarray) -> np.ndarray:
		if isinstance(possible_values[0], Iterable) and len(possible_values) > 1:
			possible_values = np.array(
				np.meshgrid(*[possible_values[i] for i in range(len(possible_values))], indexing="ij")
			).reshape(len(possible_values), -1)
		return possible_values

	def _get_possible_channeled_values(self, state: TradeState, base_currency: str, quote_currency: str) -> np.ndarray:
		return self.__state_transition_sampler.sample_next_values(state, (base_currency, quote_currency))

	def _get_possible_channel_values(self, state: TradeState, base_currency: str, quote_currency: str) -> np.ndarray:
		possible_values = self._get_possible_channeled_values(state, base_currency, quote_currency)

		possible_values = self._enumerate_channel_combinations(possible_values)
		# possible_values = self.__filter_possible_values(possible_values)

		return possible_values

	def _simulate_instrument_change_bound_mode(
			self,
			state: TradeState,
			base_currency: str,
			quote_currency: str,
			action: typing.Any
	) -> List[TradeState]:
		states = []

		possible_values = self._get_possible_channel_values(state, base_currency, quote_currency)

		for j in range(possible_values.shape[1]):
			new_state = state.__deepcopy__()
			new_state.recent_balance = state.get_agent_state().get_balance()

			new_value = np.expand_dims(possible_values[:, j], axis=1)

			new_state.get_market_state().update_state_of(
				base_currency,
				quote_currency,
				new_value
			)
			self._simulate_trades_triggers(new_state, (base_currency, quote_currency))
			states.append(new_state)

		return states

	def __simulate_instrument_change(self, state: TradeState, base_currency: str, quote_currency: str, action) -> List[TradeState]:
		if not self.__state_change_delta_model_mode:
			return self._simulate_instrument_change_bound_mode(state, base_currency, quote_currency, action)

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

	def _simulate_action(self, state: TradeState, action: Action):  # TODO: SETUP CACHER
		# state = copy.deepcopy(state)

		if action is None:
			return

		if isinstance(action, ActionSequence):
			for action in action.actions:
				self._simulate_action(state, action)
			return

		assert isinstance(action, TraderAction)

		if action.action == TraderAction.Action.CLOSE:
			state.get_agent_state().close_trades(action.base_currency, action.quote_currency)
			return
		try:
			state.get_agent_state().open_trade(
				action,
				state.get_market_state().get_current_price(action.base_currency, action.quote_currency)
			)
		except InsufficientFundsException:
			pass
