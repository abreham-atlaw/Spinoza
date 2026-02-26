import os.path
import unittest

import numpy as np

from core import Config
from core.agent.action import TraderAction
from core.agent.utils.state_predictor import TitanStatePredictor
from core.di import AgentUtilsProvider
from core.environment.trade_state import MarketState, AgentState, TradeState


class TitanStatePredictorTest(unittest.TestCase):

	def __init_state(self):
		self.instruments = [
			("AUD", "USD"),
			("USD", "ZAR")
		]
		self.channels = 4
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

	def setUp(self):
		Config.CORE_MODEL_CONFIG.path = os.path.join(Config.BASE_DIR, "temp/models/dra.zip")
		self.model = AgentUtilsProvider.provide_core_torch_model()

		self.predictor = TitanStatePredictor(
			model=self.model
		)

		self.states = [
			self.__init_state()
			for _ in range(10)
		]

	def test_predict(self):

		y = self.predictor.predict(self.states, [TraderAction("AUD", "USD", TraderAction.Action.CLOSE)] * len(self.states))

		self.assertEqual(y.shape[:-1], (len(self.states), len(self.instruments), self.channels,))
		print(y.shape)

	def test_predict_instrument(self):

		y = self.predictor.predict(self.states, [None]*len(self.states))
		y_ins = self.predictor.predict(self.states, [None]*len(self.states), instrument=self.instruments[0])

		self.assertEqual(y_ins.shape[:-1], (len(self.states), self.channels))

		self.assertTrue(np.all(y_ins == y[:, 0]))
