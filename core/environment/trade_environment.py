from typing import *
from abc import abstractmethod, ABC

from core import Config
from .trade_state import TradeState
from core.agent.action import TraderAction, Action, ActionSequence
from lib.rl.environment import Environment


class TradeEnvironment(Environment, ABC):

	def __init__(
			self,
			time_penalty=Config.TIME_PENALTY,
			trade_size_gap=Config.AGENT_TRADE_SIZE_GAP,
			market_state_memory=Config.MARKET_STATE_MEMORY
	):
		super(TradeEnvironment, self).__init__()
		self._state: Union[TradeState, None] = None
		self.__time_penalty = time_penalty
		self.__trade_size_gap = trade_size_gap
		self._market_state_memory = market_state_memory

	@abstractmethod
	def _initiate_state(self) -> TradeState:
		pass

	@abstractmethod
	def _refresh_state(self, state: TradeState = None) -> TradeState:
		pass

	def _initialize(self):
		super()._initialize()
		self._state = self._initiate_state()

	def _close_trades(self, base_currency, quote_currency):
		self.get_state().get_agent_state().close_trades(base_currency, quote_currency)

	def _open_trade(self, action: TraderAction):
		self.get_state().get_agent_state().open_trade(
			action
		)

	def get_reward(self, state: TradeState or None = None):
		if state is None:
			state = self.get_state()

		return state.get_recent_balance_change() + self.__time_penalty

	def perform_action(self, action: Action):

		if isinstance(action, ActionSequence):
			for action in action.actions:
				self.perform_action(action)
			return

		assert action is None or isinstance(action, TraderAction)

		recent_balance = self.get_state().get_agent_state().get_balance()

		if action is None:
			pass

		elif action.action == TraderAction.Action.CLOSE:
			self._close_trades(action.base_currency, action.quote_currency)

		else:
			self._open_trade(action)
		self._state = self._refresh_state()
		self._state.recent_balance = recent_balance

	def render(self):
		pass

	def update_ui(self):
		pass

	def check_is_running(self) -> bool:
		return True

	def __is_action_sequence_valid(self, sequence: ActionSequence, state: TradeState) -> bool:

		margin_available = state.get_agent_state().get_margin_available()

		for action in sequence.actions:
			if action.action == TraderAction.Action.CLOSE:
				margin_available += sum([trade.get_trade().margin_used for trade in state.get_agent_state().get_open_trades(action.base_currency, action.quote_currency)])
			else:
				margin_available -= action.margin_used
				if margin_available < 0:
					return False

		return True

	def is_action_valid(self, action: TraderAction, state: TradeState) -> bool:
		if isinstance(action, ActionSequence):
			return self.__is_action_sequence_valid(action, state)

		if action is not None and action.action != TraderAction.Action.CLOSE and state.get_agent_state().get_margin_available() < action.margin_used:
			return False
		return True  # TODO: MORE VALIDATIONS

	def get_state(self) -> TradeState:
		if self._state is None:
			raise Exception("State not Initialized.")
		return self._state

	def is_episode_over(self, state=None) -> bool:
		return False
