import os
import unittest
from copy import deepcopy

import numpy as np

from core import Config
from core.agent.utils.state_predictor import LegacyStatePredictor
from core.di import AgentUtilsProvider
from core.environment.trade_state import MarketState, AgentState, TradeState


class LegacyStatePredictorTest(unittest.TestCase):

	def __init_state(self):
		self.instruments = [
			("AUD", "USD"),
			("USD", "ZAR")
		]
		self.channels = 1
		self.memory_len = 128
		market_state = MarketState(
			tradable_pairs=self.instruments,
			memory_len=self.memory_len,
			channels=self.channels,
			currencies=["AUD", "USD", "ZAR"]
		)

		for base_currency, quote_currency in self.instruments:
			market_state.update_state_of(base_currency, quote_currency, np.random.random((self.channels, self.memory_len)))

		agent_state = AgentState(
			market_state=market_state,
			balance=100
		)

		return TradeState(
			agent_state=agent_state,
			market_state=market_state
		)

	def __init_final_state(self, state: TradeState) -> TradeState:
		new_state = deepcopy(state)
		new_state.get_market_state().update_state_of(*self.instruments[0], np.random.random((self.channels, 1)))
		return new_state

	def setUp(self):
		Config.CORE_MODEL_CONFIG.path = os.path.join(Config.BASE_DIR, "temp/models/abrehamalemu-spinoza-training-cnn-2-it-87-tot.zip")
		self.model = AgentUtilsProvider.provide_core_torch_model()
		self.predictor = LegacyStatePredictor(
			model=self.model,
			extra_len=124
		)
		self.states = [
			self.__init_state()
			for _ in range(10)
		]
		self.final_states = [
			self.__init_final_state(state)
			for state in self.states
		]

	def test_predict(self):

		y = self.predictor.predict(self.states, [None] * len(self.states), final_states=self.final_states)

		self.assertEqual(y.shape[:-1], (len(self.states),))
		print(y.shape)
