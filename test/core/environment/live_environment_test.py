import unittest

from lib.network.oanda import Trader
from core.environment import LiveEnvironment
from core.agent.action.trader_action import TraderAction
from core import Config


class LiveEnvironmentTest(unittest.TestCase):

	def setUp(self):
		self.trader = Trader(Config.OANDA_TOKEN, Config.OANDA_TRADING_ACCOUNT_ID)
		self.live_environment = LiveEnvironment()
		self.live_environment.start()

	def test_open_trade(self):
		base_currency, quote_currency = "USD", "CAD"
		self.trader.close_all_trades()
		self.live_environment.perform_action(
			TraderAction(
				base_currency,
				quote_currency,
				TraderAction.Action.SELL,
				margin_used=20
			)
		)

		open_trades = self.trader.get_open_trades()
		self.assertEqual(
			len(open_trades),
			1
		)

	def test_close_trade(self):
		base_currency, quote_currency = "USD", "CAD"
		self.trader.trade(
			(base_currency, quote_currency),
			Trader.TraderAction.BUY,
			20
		)

		open_trades = self.trader.get_open_trades()
		assert len(open_trades) >= 1

		self.live_environment.perform_action(
			TraderAction(
				base_currency,
				quote_currency,
				TraderAction.Action.CLOSE
			)
		)

		self.assertLess(
			len(self.trader.get_open_trades()),
			len(open_trades)
		)

	def test_new_state(self):
		base_currency, quote_currency = "USD", "CAD"

		old_state = self.live_environment.get_state()

		self.live_environment.perform_action(
			TraderAction(
				base_currency,
				quote_currency,
				TraderAction.Action.SELL,
				margin_used=20
			)
		)

		new_state = self.live_environment.get_state()

		self.assertIsNot(new_state, old_state)

