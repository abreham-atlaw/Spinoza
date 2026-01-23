from typing import *

from lib.rl.environment import ModelBasedState
from core.agent.action import TraderAction
from .market_state import MarketState
from .agent_state import AgentState


class TradeState(ModelBasedState):

	def __init__(
			self,
			market_state: MarketState = None,
			agent_state: AgentState = None,
			recent_balance: float = None,
			pre_computation: bool = True,
			is_running: bool = True,
	):
		self.market_state = market_state
		self.agent_state = agent_state
		self.recent_balance = recent_balance
		self.pre_computation = pre_computation
		self.__depth = 0
		self.__attached_state = {}
		self.is_running = is_running

	def get_market_state(self) -> MarketState:
		return self.market_state

	def get_agent_state(self) -> AgentState:
		return self.agent_state

	def get_recent_balance(self) -> float:
		return self.recent_balance

	def get_recent_balance_change(self) -> float:
		if self.get_recent_balance() is None:
			return 0
		return self.get_agent_state().get_balance() - self.get_recent_balance()

	def set_depth(self, depth: int):
		self.__depth = depth

	def get_depth(self) -> int:
		return self.__depth

	def attach_state(self, key: Hashable, state: Any):
		self.__attached_state[key] = state

	def detach_state(self, key: Hashable) -> Any:
		return self.__attached_state.pop(key)

	def get_attached_state(self, key: Hashable) -> Any:
		return self.__attached_state[key]

	def is_state_attached(self, key: Hashable) -> bool:
		return key in self.__attached_state.keys()

	def end_episode(self):
		self.is_running = False

	def __deepcopy__(self, memo=None):
		market_state = self.market_state.__deepcopy__()
		agent_state = self.agent_state.__deepcopy__(memo={'market_state': market_state})

		return TradeState(market_state, agent_state, self.get_recent_balance(), is_running=self.is_running)

	def __hash__(self):
		return hash((self.market_state, self.agent_state, self.get_recent_balance(), self.is_running))

	def __eq__(self, other):
		if not isinstance(other, TraderAction):
			return False
