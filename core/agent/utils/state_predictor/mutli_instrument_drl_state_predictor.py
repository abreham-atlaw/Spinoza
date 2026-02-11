import typing

import numpy as np

from core.environment.trade_state import TradeState, AgentState
from .multi_instrument_state_predictor import MultiInstrumentPredictor
from core.agent.action import Action, TraderAction, ActionSequence


class MultiInstrumentDRLStatePredictor(MultiInstrumentPredictor):

	__open_trade_encode_size = 7
	__action_encode_size = 5

	def __init__(self, *args, **kwargs):
		super().__init__(*args, **kwargs)

	@staticmethod
	def __encode_instrument(instrument: typing.Tuple[str, str], state: TradeState) -> int:
		return state.get_market_state().get_tradable_pairs().index(instrument)

	@staticmethod
	def __encode_trade_action(action: int) -> int:
		return 0 if action == TraderAction.Action.CLOSE else 1 if action == TraderAction.Action.BUY else -1

	def __encode_open_trade(self, trade: AgentState.OpenTrade, state: TradeState) -> np.ndarray:
		return np.array([
			trade.get_trade().margin_used / state.get_agent_state().get_balance(), # MARGIN USED
			self.__encode_trade_action(trade.get_trade().action), # DIRECTION
			np.log(trade.get_current_value() / trade.get_enter_value()), # CURRENT_VALUE / ENTER_VALUE
			trade.get_unrealized_profit() / state.get_agent_state().get_balance(), # UNREALIZED PL
			trade.get_trade().stop_loss if trade.get_trade().stop_loss is not None else 0, # STOP LOSS
			trade.get_trade().take_profit if trade.get_trade().take_profit is not None else 0,  # TAKE PROFIT
			self.__encode_instrument(
				(trade.get_trade().base_currency, trade.get_trade().quote_currency),
				state
			), # INSTRUMENT
		])

	@staticmethod
	def __get_channel_size(state: TradeState) -> int:
		return state.get_market_state().channels * len(state.get_market_state().get_tradable_pairs())

	def __encode_state(self, state: TradeState) -> np.ndarray:
		if len(state.get_agent_state().get_open_trades()) == 0:
			return np.zeros((self.__get_channel_size(state), self.__open_trade_encode_size))

		x = np.stack([
			self.__encode_open_trade(trade, state)
			for trade in state.get_agent_state().get_open_trades()
		])
		x = np.concatenate([
			x,
			np.zeros((self.__get_channel_size(state) - x.shape[0], self.__open_trade_encode_size))
		])
		return x

	def __encode_states(self, states: typing.List[TradeState]):
		return np.stack([
			self.__encode_state(state)
			for state in states
		], axis=0)

	def __encode_action(self, action: Action, state: TradeState) -> np.ndarray:
		if action is None:
			return np.zeros((self.__get_channel_size(state), self.__action_encode_size))

		if isinstance(action, ActionSequence):
			action = list(filter(
				lambda a: isinstance(a, TraderAction) and a.action != TraderAction.Action.CLOSE,
				action.actions
			))[-1]

		assert isinstance(action, TraderAction)

		x = np.array([
			action.margin_used / state.get_agent_state().get_balance(),
			self.__encode_trade_action(action.action),
			action.stop_loss if action.stop_loss is not None else 0,
			action.take_profit if action.take_profit is not None else 0,
			self.__encode_instrument((action.base_currency, action.quote_currency), state)
		])

		x = np.concatenate([
			np.expand_dims(x, axis=0),
			np.zeros((self.__get_channel_size(state) - 1, self.__open_trade_encode_size))
		])
		return x

	def __encode_actions(self, actions: typing.List[Action], states: typing.List[TradeState]) -> np.ndarray:
		return np.stack([
			self.__encode_action(action, state)
			for action, state in zip(actions, states)
		])

	def prepare_input(
			self,
			states: typing.List[TradeState],
			actions: typing.List[Action],
			*args, **kwargs
	) -> np.ndarray:
		time_series = super().prepare_input(states, actions, *args, **kwargs)
		extra_state = self.__encode_states(states)
		action = self.__encode_actions(actions, states)

		x = np.concatenate([
			time_series, extra_state, action
		], axis=-1)

		return x
