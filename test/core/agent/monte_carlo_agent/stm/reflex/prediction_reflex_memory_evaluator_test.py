import unittest

import numpy as np

from core.agent.action import TraderAction
from core.agent.agents.montecarlo_agent.stm.reflex import TraderReflexMemoryEvaluator
from core.di import AgentUtilsProvider
from core.environment.trade_state import MarketState, AgentState, TradeState
from lib.utils.stm import StochasticShortTermMemory


class TraderReflexMemoryEvaluatorTest(unittest.TestCase):

	def __init_state(self, m: float, b: float, t: int):
		channels = 4
		market_state = MarketState(
			currencies=["AUD", "USD"],
			memory_len=128,
			channels=channels,
			tradable_pairs=[("AUD", "USD"), ("USD", "AUD")]
		)
		market_state.update_state_of("AUD", "USD", np.arange(128*channels).reshape((channels, -1)))
		market_state.update_state_of("AUD", "USD", np.array([[128*m] for _ in range(channels)]))


		agent_state = AgentState(
			market_state=market_state,
			balance=100*b
		)
		for i in range(t):
			agent_state.open_trade(agent_state.rectify_action(TraderAction("AUD", "USD", TraderAction.Action.BUY, 20)))
		return TradeState(
			market_state, agent_state
		)

	def setUp(self):
		self.evaluator = AgentUtilsProvider.provide_reflex_memory_evaluator()
		self.states = [
			self.__init_state(1.1, 1.1, 1),
			self.__init_state(0.9, 1.1, 1),
			self.__init_state(0.9, 1.1, 0),
			self.__init_state(1.1, 1.1, 0),
		]

		self.appox_state = self.__init_state(1.05, 1.1, 1)

	def test_evaluate(self):

		losses = [
			self.evaluator(self.appox_state, state)
			for state in self.states
		]
		print(losses)

		self.assertLess(losses[0], losses[1])
		self.assertLess(losses[1], losses[2])
		self.assertLess(losses[3], losses[2])

	def test_stm(self):

		stm = StochasticShortTermMemory(evaluator=self.evaluator, size=5)
		for state in self.states:
			stm.memorize(state)

		recalled = stm.recall(self.appox_state)
		self.assertIs(recalled, self.states[0])
