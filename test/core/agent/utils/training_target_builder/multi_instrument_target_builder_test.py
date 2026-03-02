import os
import unittest

import numpy as np

from core import Config
from core.agent.action import TraderAction
from core.agent.utils.training_target_builder import MultiInstrumentTargetBuilder
from core.environment.trade_state import MarketState, AgentState, TradeState
from lib.utils.fileio import load_json


class MultiInstrumentTargetBuilderTest(unittest.TestCase):

	def __init_state(self):

		channels = len(self.channels)
		memory_len = 128
		market_state = MarketState(
			tradable_pairs=self.instruments,
			memory_len=memory_len,
			channels=channels,
			currencies=["AUD", "USD", "ZAR"]
		)

		for base_currency, quote_currency in self.instruments:
			market_state.update_state_of(base_currency, quote_currency, np.random.random((channels, memory_len)))

		agent_state = AgentState(
			market_state=market_state,
			balance=100
		)

		return TradeState(
			agent_state=agent_state,
			market_state=market_state
		)

	def setUp(self):
		self.channels = ('c', 'l', 'h', 'o')
		self.instruments = [
			("AUD", "USD"),
			("USD", "ZAR")
		]
		self.bounds = load_json(os.path.join(Config.RES_DIR, "bounds/15.json"))
		self.builder = MultiInstrumentTargetBuilder(
			channels=self.channels,
			bounds=self.bounds
		)

	def test_build_target(self):
		state = self.__init_state()
		final_state = self.__init_state()
		action = TraderAction(
			base_currency="AUD",
			quote_currency="USD",
			action=TraderAction.Action.BUY,
			margin_used=70.0
		)
		value = 2.0

		y = self.builder.build(state, action, final_state, value)

		self.assertEqual(y[0, -1], value / state.get_agent_state().get_balance())
		self.assertEqual(y.shape, (len(self.channels)*len(self.instruments), len(self.bounds)+2))
