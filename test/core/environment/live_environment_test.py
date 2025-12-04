import unittest

import matplotlib.pyplot as plt

from lib.network.oanda import Trader
from core.environment import LiveEnvironment
from core.agent.action.trader_action import TraderAction
from core import Config


class LiveEnvironmentTest(unittest.TestCase):

	def setUp(self):
		Config.AGENT_STATIC_INSTRUMENTS = [
			("AUD", "USD")
		]
		Config.AGENT_MA_WINDOW_SIZE = 32
		Config.MARKET_STATE_MEMORY = 128
		Config.MARKET_STATE_CHANNELS = (
				"c", "o", "l", "h", "v",
				"time.year", "time.month", "time.day", "time.hour", "time.minute", "time.second"
		)
		Config.MARKET_STATE_SMOOTHED_CHANNELS = ("c",)

		self.trader = Trader(Config.OANDA_TOKEN, Config.OANDA_TRADING_ACCOUNT_ID)
		Config.MARKET_STATE_MEMORY = 128
		self.live_environment = LiveEnvironment()
		self.live_environment.start()

	def test_open_trade(self):
		base_currency, quote_currency = Config.AGENT_STATIC_INSTRUMENTS[0]
		self.trader.close_all_trades()
		self.live_environment.perform_action(
			TraderAction(
				base_currency,
				quote_currency,
				TraderAction.Action.SELL,
				margin_used=20,
				stop_loss=1.0
			)
		)

		open_trades = self.trader.get_open_trades()
		self.assertEqual(
			len(open_trades),
			1
		)

	def test_close_trade(self):
		base_currency, quote_currency = Config.AGENT_STATIC_INSTRUMENTS[0]
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
		base_currency, quote_currency = Config.AGENT_STATIC_INSTRUMENTS[0]

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

		for instrument in Config.AGENT_STATIC_INSTRUMENTS:

			for i in range(len(Config.MARKET_STATE_CHANNELS)):
				plt.figure()
				plt.title(f"Channel: {Config.MARKET_STATE_CHANNELS[i]}")
				for state in [old_state, new_state]:
					plt.plot(state.get_market_state().get_channels_state(*instrument,)[i])
		plt.show()
		self.assertIsNot(new_state, old_state)

