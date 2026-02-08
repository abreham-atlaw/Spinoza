import unittest

import numpy as np

from core import Config
from core.agent.utils.state_transition_sampler import BasicStateTransitionSampler
from core.environment.trade_state import MarketState, TradeState, AgentState


class BasicStateTransitionSamplerTest(unittest.TestCase):

	def setUp(self):
		self.channels = ("c", "l", "h", "o")
		self.sampler = BasicStateTransitionSampler(
			bounds=Config.AGENT_STATE_CHANGE_DELTA_STATIC_BOUND,
			channels=self.channels,
			simulated_channels=self.channels
		)

	def test_sample_next_values(self):
		SEQ_LEN = 10
		market_state = MarketState(
			currencies=["AUD", "USD"],
			tradable_pairs=[
				("AUD", "USD")
			],
			memory_len=SEQ_LEN,
			channels=len(self.channels)
		)
		market_state.update_state_of("AUD", "USD", np.random.random((len(self.channels), 5)))

		state = TradeState(
			market_state=market_state,
			agent_state=AgentState(
				balance=100,
				market_state=market_state,
			)
		)

		values = self.sampler.sample_next_values(
			state, ("AUD", "USD")
		)

		print(values)
