from typing import *
from abc import abstractmethod, ABC

from core import Config
from lib.utils.logger import Logger
from .trade_state import TradeState
from core.agent.action import TraderAction, Action, ActionSequence, EndEpisode
from lib.rl.environment import Environment


class TradeEnvironment(Environment, ABC):

	def __init__(
			self,
			time_penalty=Config.TIME_PENALTY,
			trade_size_gap=Config.AGENT_TRADE_SIZE_GAP,
			market_state_memory=Config.MARKET_STATE_MEMORY,
			loss_weight: float = Config.AGENT_LOSS_WEIGHT
	):
		super(TradeEnvironment, self).__init__()
		self._state: Union[TradeState, None] = None
		self.__time_penalty = time_penalty
		self.__trade_size_gap = trade_size_gap
		self._market_state_memory = market_state_memory
		self.__is_active = True
		self.__loss_weight = loss_weight
		Logger.info(f"Initializing {self.__class__.__name__} with loss_weight={loss_weight}")

	@abstractmethod
	def _initiate_state(self) -> TradeState:
		pass

	@abstractmethod
	def _refresh_state(self, state: TradeState = None) -> TradeState:
		pass

	def _initialize(self):
		super()._initialize()
		self._state = self._initiate_state()

	def _end_episode(self):
		self._close_all_trades()
		self.get_state().is_running = False
		self.__is_active = False

	def _close_trades(self, base_currency, quote_currency):
		self.get_state().get_agent_state().close_trades(base_currency, quote_currency)

	def _close_all_trades(self):
		for instrument in set([(trade.get_trade().base_currency, trade.get_trade().quote_currency) for trade in self.get_state().get_agent_state().get_open_trades()]):
			self._close_trades(*instrument)

	def _open_trade(self, action: TraderAction):
		self.get_state().get_agent_state().open_trade(
			action
		)

	def get_reward(self, state: TradeState or None = None):
		if state is None:
			state = self.get_state()

		reward = state.get_recent_balance_change()
		if reward < 0:
			reward *= self.__loss_weight

		return reward + self.__time_penalty

	def _perform_action(self, action: Action):

		if isinstance(action, EndEpisode):
			self._end_episode()
			return

		if isinstance(action, ActionSequence):
			for action in action.actions:
				self._perform_action(action)
			return

		assert action is None or isinstance(action, TraderAction)

		if action is None:
			pass

		elif action.action == TraderAction.Action.CLOSE:
			self._close_trades(action.base_currency, action.quote_currency)

		else:
			self._open_trade(action)

	def perform_action(self, action: Action):

		recent_balance = self.get_state().get_agent_state().get_balance()

		self._perform_action(action)

		self._state = self._refresh_state()
		self._state.recent_balance = recent_balance

	def render(self):
		pass

	def update_ui(self):
		pass

	def check_is_running(self) -> bool:
		return self.get_state().is_running

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

		if isinstance(action, EndEpisode):
			return True

		if action is not None and action.action != TraderAction.Action.CLOSE and state.get_agent_state().get_margin_available() < action.margin_used:
			return False
		return True  # TODO: MORE VALIDATIONS

	def get_state(self) -> TradeState:
		if self._state is None:
			raise Exception("State not Initialized.")
		return self._state

	def is_episode_over(self, state=None) -> bool:
		state = state if state is not None else self.get_state()
		return not state.is_running

	def get_latest_state(self):
		return self._refresh_state()