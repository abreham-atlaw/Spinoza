import typing

import numpy as np

from core.environment.trade_state import TradeState, AgentState, MarketState
from lib.utils.stm import StochasticMemoryEvaluator


class TraderReflexMemoryEvaluator(StochasticMemoryEvaluator):

	@staticmethod
	def __evaluate_agent_state(state0: AgentState, state1: AgentState) -> float:
		return (
			abs(state0.get_balance() - state1.get_balance()) +
			abs(len(state0.get_open_trades()) - len(state1.get_open_trades()))*100
		)

	@staticmethod
	def _evaluate_market_state(state0: TradeState, state1: TradeState) -> float:
		state0, state1 = state0.get_market_state(), state1.get_market_state()
		return np.sum([
			np.abs(state0.get_channels_state(base_currency, quote_currency) - state1.get_channels_state(base_currency, quote_currency))
			for base_currency, quote_currency in state0.get_tradable_pairs()
		])

	def evaluate(self, cue: TradeState, memory: TradeState) -> float:
		return self.__evaluate_agent_state(cue.get_agent_state(), memory.get_agent_state()) + \
			self._evaluate_market_state(cue, memory)
