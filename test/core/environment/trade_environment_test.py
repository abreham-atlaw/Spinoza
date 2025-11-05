import unittest
from unittest import mock

from core.environment import TradeEnvironment
from core.environment.trade_state import TradeState
from core.agent.action.trader_action import TraderAction


class TradeEnvironmentTest(unittest.TestCase):

	class TestTradeEnvironment(TradeEnvironment):

		def __init__(self, mock_state, *args, **kwargs):
			super().__init__(*args, **kwargs)
			self.mock_state = mock_state

		def _refresh_state(self, state: TradeState) -> TradeState:
			pass

		def _initiate_state(self) -> TradeState:
			return self.mock_state

	def setUp(self):
		self.mock_market_state: mock.Mock() = mock.Mock()
		self.mock_agent_state: mock.Mock() = mock.Mock()
		self.mock_state = mock.Mock()
		self.mock_state.get_market_state.get_return = self.mock_market_state
		self.mock_state.get_agent_state.get_return = self.mock_agent_state

		self.environment = TradeEnvironmentTest.TestTradeEnvironment(self.mock_state)
		self.environment.start()

	def test_perform_action(self):
		ta = TraderAction(
				"USD",
				"EUR",
				TraderAction.Action.SELL,
				margin_used=20,
				units=1000
		)
		self.environment.perform_action(ta)
		self.mock_agent_state.open_trade.assert_called_with(ta)

	def test_get_valid_action(self):
		tp = [
			("USD", "EUR"),
			("AUD", "GBP"),
			("AUD", "USD"),
			("GBP", "CAD")
		]
		self.mock_market_state.get_tradable_pairs.get_return = tp
		self.mock_agent_state.get_balance.get_return = 100

		open_trades = []
		for i in range(3):
			trade = mock.Mock()
			trade.get_trade.base_currency, trade.get_trade.quote_currency = tp[i]
			open_trades.append(trade)

		self.mock_agent_state.get_open_trades.get_return = open_trades

		self.assertEqual(
			len(self.environment.get_valid_actions()),
			164
		)



