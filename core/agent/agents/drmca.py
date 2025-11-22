import random
import typing
from abc import ABC

import numpy as np

from core import Config
from core.agent.agents.dnn_transition_agent import TraderDNNTransitionAgent
from core.agent.concurrency.mc.data.serializer import TraderNodeSerializer
from core.agent.action import TraderAction, Action, ActionSequence
from core.agent.utils.cache import Cache
from core.di import AgentUtilsProvider
from core.environment.trade_state import TradeState, AgentState
from core.utils.research.model.model.utils import TransitionOnlyModel
from core.utils.research.model.model.utils import WrappedModel

from core.utils.research.model.model.utils import TemperatureScalingModel
from core.utils.research.model.model.utils.cached_model import CachedModel
from lib.rl.agent.drmca import DeepReinforcementMonteCarloAgent
from lib.rl.agent.dta import Model, TorchModel
from lib.rl.environment import ModelBasedState
from lib.utils.torch_utils.model_handler import ModelHandler


class TraderDeepReinforcementMonteCarloAgent(DeepReinforcementMonteCarloAgent, TraderDNNTransitionAgent, ABC):

	__OPEN_TRADE_ENCODE_SIZE = 6

	def __init__(
			self,
			*args,
			batch_size=Config.UPDATE_EXPORT_BATCH_SIZE,
			train=Config.UPDATE_TRAIN,
			save_path=Config.UPDATE_SAVE_PATH,
			encode_max_open_trade=Config.AGENT_MAX_OPEN_TRADES,
			wp=Config.AGENT_DRMCA_WP,
			top_k_nodes=Config.AGENT_TOP_K_NODES,
			dynamic_k_threshold=Config.AGENT_DYNAMIC_K_THRESHOLD,
			dump_nodes=Config.AGENT_DUMP_NODES,
			dump_path=Config.AGENT_DUMP_NODES_PATH,
			dump_visited_only=Config.AGENT_DUMP_VISITED_ONLY,
			discount=Config.AGENT_DISCOUNT_FACTOR,
			discount_function=Config.AGENT_DISCOUNT_FUNCTION,
			use_transition_only_model=Config.AGENT_MODEL_USE_TRANSITION_ONLY,
			use_extra_data: bool = Config.AGENT_USE_EXTRA_DATA,
			**kwargs
	):
		self.__use_transition_only = use_transition_only_model
		super().__init__(
			*args,
			batch_size=batch_size,
			train=train,
			save_path=save_path,
			wp=wp,
			top_k_nodes=top_k_nodes,
			dynamic_k_threshold=dynamic_k_threshold,
			dump_nodes=dump_nodes,
			dump_path=dump_path,
			dump_visited_only=dump_visited_only,
			node_serializer=TraderNodeSerializer(),
			discount=discount,
			discount_function=discount_function,
			state_repository=AgentUtilsProvider.provide_state_repository(),
			**kwargs
		)
		self.__encode_max_open_trades = encode_max_open_trade
		self.__dra_input_cache = Cache()
		self.__use_extra_data = use_extra_data

	def _init_model(self) -> Model:
		model = TemperatureScalingModel(
			model=ModelHandler.load(Config.CORE_MODEL_CONFIG.path),
			temperature=Config.AGENT_MODEL_TEMPERATURE
		)
		print(f"Using Temperature: {Config.AGENT_MODEL_TEMPERATURE}")
		if Config.AGENT_MODEL_USE_CACHED_MODEL:
			model = CachedModel(
				model=model,
			)
		if Config.AGENT_MODEL_USE_TRANSITION_ONLY:
			model = TransitionOnlyModel(
				model=model,
				extra_len=Config.AGENT_MODEL_EXTRA_LEN
			)
		return TorchModel(
				WrappedModel(
					model,
					seq_len=Config.MARKET_STATE_MEMORY,
					window_size=Config.AGENT_MA_WINDOW_SIZE,
					use_ma=Config.AGENT_USE_SMOOTHING,
				)
			)

	@property
	def _transition_model(self) -> Model:
		return self._model

	@staticmethod
	def __encode_action(state: TradeState, action: typing.Optional[TraderAction]) -> np.ndarray:
		encoded = np.zeros((4,))
		if action is None:
			return encoded
		encoded[action.action] = 1
		if action.action != TraderAction.Action.CLOSE:
			encoded[3] = state.agent_state.calc_required_margin(action.units, action.base_currency, action.quote_currency)/state.agent_state.get_margin_available()
		return encoded

	def __encode_open_trade(self, trade: AgentState.OpenTrade, state: TradeState) -> np.ndarray:
		encoded = np.zeros((self.__OPEN_TRADE_ENCODE_SIZE,))
		encoded[:2] = [trade.get_enter_value(), trade.get_current_value()]
		encoded[trade.get_trade().action + 2] = 1
		encoded[4] = trade.get_trade().margin_used / state.agent_state.get_balance()
		encoded[5] = state.agent_state.to_agent_currency(trade.get_unrealized_profit(), trade.get_trade().quote_currency) / state.get_agent_state().get_balance()
		return encoded

	def __encode_open_trades(self, state: TradeState):
		open_trades = state.agent_state.get_open_trades()
		if len(open_trades) > self.__encode_max_open_trades:
			raise ValueError("Found more open trades than `encode_max_open_trades`")
		encoded = np.zeros((self.__OPEN_TRADE_ENCODE_SIZE * self.__encode_max_open_trades))
		for i, trade in enumerate(open_trades):
			encoded[i*self.__OPEN_TRADE_ENCODE_SIZE: (i+1)*self.__OPEN_TRADE_ENCODE_SIZE] = self.__encode_open_trade(trade, state)
		return encoded

	def __prepare_model_input(
			self,
			state: TradeState,
			action: typing.Optional[Action],
			target_instrument: typing.Tuple[str, str]
	) -> np.ndarray:

		def prepare_model_input(state: TradeState, action: typing.Optional[Action], target_instrument: typing.Tuple[str, str]) -> np.ndarray:

			# TODO: ENCODE ALL ACTIONS
			if isinstance(action, ActionSequence):
				action = action.actions[0]

			market_data = state.get_market_state().get_channels_state(target_instrument[0], target_instrument[1])

			extra_data = np.repeat(
				np.expand_dims(
					np.concatenate((
						self.__encode_action(state, action),
						self.__encode_open_trades(state)
					), axis=-1),
					axis=0
				),
				axis=0,
				repeats=market_data.shape[0]
			)
			data = market_data

			if self.__use_extra_data:
				data = np.concatenate((market_data, extra_data), axis=-1)

			if not self._use_multi_channels:
				assert data.shape[0] == 1
				data = np.squeeze(data, axis=0)

			return data

		return self.__dra_input_cache.cached_or_execute((state, action, target_instrument), lambda: prepare_model_input(state, action, target_instrument))

	def _prepare_single_dta_input(self, state: TradeState, action: Action, final_state: TradeState) -> np.ndarray:
		return self.__prepare_model_input(state, action, self._get_target_instrument(state, action, final_state))

	def _prepare_dra_input(self, state: TradeState, action: Action) -> np.ndarray:
		if isinstance(action, ActionSequence):  # TODO: ENCODE ALL ACTIONS
			action = action.actions[0]

		if action is None:
			instrument = random.choice(state.get_market_state().get_tradable_pairs())
		else:
			instrument = action.base_currency, action.quote_currency
		return self.__prepare_model_input(state, action, instrument)

	@staticmethod
	def _parse_model_output(output: np.ndarray) -> typing.Tuple[np.ndarray, float]:
		probability_distribution = output[:-1]
		value = output[-1]
		return probability_distribution, value

	def _prepare_dra_output(self, state: TradeState, action: Action, output: np.ndarray) -> float:
		_, value = self._parse_model_output(output)
		return value * state.get_agent_state().get_balance()

	def _single_prediction_to_transition_probability_bound_mode(
			self,
			initial_state: TradeState,
			output: np.ndarray,
			final_state: TradeState
	) -> float:
		output, _ = self._parse_model_output(output)
		return super()._single_prediction_to_transition_probability_bound_mode(initial_state, output, final_state)

	def _prepare_dra_train_output(
			self,
			state: TradeState,
			action: Action,
			final_state: TradeState,
			value: float
	) -> np.ndarray:
		instrument = self._get_target_instrument(state, action, final_state)
		percentage = final_state.market_state.get_current_price(*instrument)/state.market_state.get_current_price(*instrument)
		bound_idx = self._find_gap_index(percentage)
		output = np.zeros(len(self._state_change_delta_bounds)+2)
		output[bound_idx] = 1
		output[-1] = value/state.agent_state.get_balance()
		return output

	def _update_state_action_value(self, initial_state: ModelBasedState, action, final_state: ModelBasedState, value):
		DeepReinforcementMonteCarloAgent._update_state_action_value(self, initial_state, action, final_state, value)
