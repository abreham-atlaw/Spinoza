import os
from typing import *

import unittest
from unittest import mock

import numpy as np
import matplotlib.pyplot as plt

from copy import deepcopy

import torch

from core import Config
from core.environment.live_environment import LiveEnvironment, MarketState, AgentState, TradeState, TraderAction
from core.agent.agents import TraderMonteCarloAgent, TraderAgent
from core.utils.research.data.prepare.smoothing_algorithm import MovingAverage
from core.utils.research.utils.analysis.session_analyzer import SessionAnalyzer
from lib.rl.agent.dta import TorchModel
from temp import stats


class TraderAgentTest(unittest.TestCase):

	def _init_market_state(self):
		self.memory_len = 128
		self.channels = Config.MARKET_STATE_CHANNELS
		market_state = MarketState(
			currencies=["USD", "EUR", "AUD"],
			tradable_pairs=[
				("USD", "EUR"),
				("AUD", "EUR"),
				("AUD", "USD")
			],
			memory_len=self.memory_len,
			channels=len(self.channels),
		)

		for instrument in market_state.get_tradable_pairs():
			state = np.random.random((len(Config.MARKET_STATE_CHANNELS), self.memory_len))

			for channel, p in [("h", 1.1), ("l", 0.9)]:
				if channel in Config.MARKET_STATE_CHANNELS:
					state[Config.MARKET_STATE_CHANNELS.index(channel)] = state[Config.MARKET_STATE_CHANNELS.index("c")] * p
			market_state.update_state_of(*instrument, state)
		return market_state

	def __init_state(self):
		market_state = self._init_market_state()
		agent_state = AgentState(
			100,
			market_state,
		)
		return TradeState(
			market_state, agent_state
		)

	def setUp(self):
		self.agent = TraderAgent()
		self.environment = mock.Mock()
		self.agent.set_environment(self.environment)

		self.market_state = self._init_market_state()
		self.state = self.__init_state()

	def test_prediction_to_transition_probability(self):
		initial_state = self.state
		final_state = deepcopy(initial_state)
		final_state.get_market_state().update_state_of(
			"AUD", "USD",
			initial_state.get_market_state().get_channels_state("AUD", "USD")[:, -1:]*1.0001
		)
		final_state.get_market_state().get_channels_state("USD", "EUR")

		result = self.agent._single_prediction_to_transition_probability_bound_mode(
			initial_state,
			np.random.random((3, 66)),
			final_state
		)

		self.assertEqual(result, 1-output)

	def test_get_train_output(self):
		market_state = MarketState(
			currencies=["USD", "EUR", "AUD"],
			tradable_pairs=[
				("USD", "EUR"),
				("AUD", "EUR"),
				("USD", "AUD")
			],
			memory_len=5
		)
		market_state.update_state_of("USD", "EUR", np.arange(5, 10))
		market_state.update_state_of("AUD", "EUR", np.arange(1, 5))
		market_state.update_state_of("USD", "AUD", np.arange(7, 12))
		initial_state = mock.Mock()
		initial_state.get_market_state.get_return = market_state

		final_market_state = deepcopy(market_state)
		final_market_state.update_state_of("AUD", "EUR", np.array([0.5]))
		final_state = mock.Mock()
		final_state.get_market_state.get_return = final_market_state

		result = self.agent._get_train_output(
			initial_state,
			None,
			final_state
		)

		self.assertEqual(result, np.array([0]))

	def test_get_possible_states(self):

		initial_balance = 100
		agent_state = AgentState(initial_balance, self.market_state)

		state = TradeState(self.market_state, agent_state)
		state.get_agent_state().open_trade(
			state.get_agent_state().rectify_action(
				TraderAction("AUD", "USD", TraderAction.Action.BUY, margin_used=40, take_profit=1.0002))
		)

		result = self.agent._get_possible_states(
			state,
			None
		)

		self.assertTrue(
			0 in [
				len(s.get_agent_state().get_open_trades())
				for s in result
			]
		)

		self.assertEqual(
			len(result),
			3*2
		)

		for p_state in result:
			self.assertGreater(
				p_state.agent_state.get_balance(),
				initial_balance
			)
			self.assertEqual(
				len(p_state.agent_state.get_open_trades()),
				0
			)

	def test_generate_actions(self):

		market_state = MarketState(
			currencies=["USD", "EUR", "AUD"],
			tradable_pairs=[
				("USD", "EUR"),
				("AUD", "EUR"),
				("USD", "AUD")
			],
			memory_len=5
		)

		agent_state = AgentState(
			balance=100,
			market_state=market_state
		)

		state = TradeState(market_state, agent_state)

		actions = self.agent._generate_actions(state)
		print(actions)


	def test_perform_timestep(self):
		environment = LiveEnvironment()
		environment.start()
		self.agent.set_environment(environment)
		self.agent.perform_timestep()

	def test_loop(self):
		environment = LiveEnvironment()
		environment.start()
		self.agent.set_environment(environment)
		self.agent.loop()

	def test_resume_mca(self):

		def get_node(root, path):
			path = path.copy()
			node = root
			while len(path) > 0:
				node = node.get_children()[path.pop(0)]
			return node

		PATH = None

		environment = LiveEnvironment()
		environment.start()

		agent = TraderAgent()
		agent.set_environment(environment)

		node, repo = stats.load_node_repo("/home/abrehamatlaw/Downloads/Compressed/results_10/graph_dumps/1736044405.169263")
		if PATH is not None:
			node = get_node(node, path=PATH)

		state = repo.retrieve(node.id)

		plt.figure()
		agent._monte_carlo_tree_search(state)

		x = 1

	def test_get_expected_transition_probability_distribution(self):

		analyzer = SessionAnalyzer(
			session_path=os.path.join(Config.BASE_DIR, "temp/session_dumps/00/"),
			smoothing_algorithms=[
				MovingAverage(64)
			],
			instruments=[
				("AUD", "USD"),
				("USD", "ZAR")
			],
			model_key="176"
		)

		node, repo = analyzer.load_node(1)

		initial_state = repo.retrieve(node.id)
		action = node.get_children()[3].action
		final_states = [repo.retrieve(node.id) for node in node.get_children()[3].get_children()]
		initial_states = [initial_state for _ in final_states]
		actions = [action for _ in final_states]

		distribution = self.agent._get_expected_transition_probability_distribution(
			initial_states, actions, final_states
		)

		plt.scatter([state.get_market_state().get_current_price("USD","ZAR") for state in final_states], distribution)
		plt.show()
