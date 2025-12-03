import typing
from typing import *

import unittest
import requests_mock

from dataclasses import dataclass
from datetime import datetime

from core import Config
from lib.network.oanda import Trader
from lib.network.oanda.data.models import AccountSummary, Trade, CandleStick


class TraderTest(unittest.TestCase):

	class MockResponses:

		OPEN_TRADES0 = '{"trades":[{"id":"71","instrument":"GBP_USD","price":"1.36473","openTime":"2021-10-11T08:15:04.704059641Z","initialUnits":"-1000","initialMarginRequired":"40.9440","state":"OPEN","currentUnits":"-1000","realizedPL":"0.0000","financing":"0.0000","dividendAdjustment":"0.0000","unrealizedPL":"-0.0900","marginUsed":"40.9416","takeProfitOrder":{"id":"72","createTime":"2021-10-11T08:15:04.704059641Z","type":"TAKE_PROFIT","tradeID":"71","price":"1.36268","timeInForce":"GTC","triggerCondition":"DEFAULT","state":"PENDING"},"stopLossOrder":{"id":"73","createTime":"2021-10-11T08:15:04.704059641Z","type":"STOP_LOSS","tradeID":"71","price":"1.36678","timeInForce":"GTC","triggerCondition":"DEFAULT","triggerMode":"TOP_OF_BOOK","state":"PENDING"},"trailingStopLossOrder":{"id":"74","createTime":"2021-10-11T08:15:04.704059641Z","type":"TRAILING_STOP_LOSS","tradeID":"71","distance":"0.00200","timeInForce":"GTC","triggerCondition":"DEFAULT","triggerMode":"TOP_OF_BOOK","state":"PENDING","trailingStopValue":"1.36682"}},{"id":"60","instrument":"EUR_USD","price":"1.15960","openTime":"2021-10-05T20:21:51.973663137Z","initialUnits":"-100","initialMarginRequired":"2.3193","state":"OPEN","currentUnits":"-100","realizedPL":"0.0000","financing":"-0.0033","dividendAdjustment":"0.0000","unrealizedPL":"0.2120","marginUsed":"2.3148"}],"lastTransactionID":"74"}'
		

	@dataclass
	class TestTradeContainer:

		instrument: Tuple[str, str]
		action: int
		margin: int
		
		proper_instrument: Tuple[str, str]

	TRADES = [
		TestTradeContainer(
			("USD", "EUR"),
			Trader.TraderAction.SELL,
			30,
			("EUR", "USD")
		),

		TestTradeContainer(
			("JPY", "GBP"),
			Trader.TraderAction.BUY,
			30,
			("GBP", "JPY")
		),

		TestTradeContainer(
			("AUD", "CAD"),
			Trader.TraderAction.SELL,
			20,
			("AUD", "CAD")
		),
	]



	def setUp(self) -> None:
		self.trader = Trader(
			Config.OANDA_TOKEN,
			Config.OANDA_TEST_ACCOUNT_ID
		)
		self.trader.close_all_trades()

	def tearDown(self):
		# self.trader.close_all_trades()
		pass

	def test_get_account_summary(self):
	
		summary: AccountSummary  = self.trader.get_account_summary()
		self.assertIsInstance(summary, AccountSummary)
		self.assertEqual(summary.id, Config.OANDA_TEST_ACCOUNT_ID)

	@requests_mock.Mocker()
	def test_get_open_trades(self, m: requests_mock.Mocker):
		m.get(
			f"https://api-fxpractice.oanda.com/v3/accounts/{Config.OANDA_TEST_ACCOUNT_ID}/openTrades/",
			text=TraderTest.MockResponses.OPEN_TRADES0
		)

		open_trades: List[Trade] = self.trader.get_open_trades()
		self.assertEqual(len(open_trades), 2)
		self.assertEqual(open_trades[0].get_action(), Trader.TraderAction.SELL)
		self.assertEqual(open_trades[0].id, '71')
	
	def test_trade(self):
		
		response = self.trader.trade(
			TraderTest.TRADES[0].instrument,
			TraderTest.TRADES[0].action,
			TraderTest.TRADES[0].margin
		)
		self.assertTrue(response.is_successful())
		self.assertIn(response.tradeOpened.tradeID, [trade.id for trade in self.trader.get_open_trades()])

	def test_get_price(self):
		response = self.trader.get_price(("USD", "JPY"))
		self.assertGreater(response, 1)

	def test_close_trades(self):
		base_currency, quote_currency = "CAD", "USD"
		self.trader.trade((base_currency, quote_currency), Trader.TraderAction.BUY, 20)
		open_trades = self.trader.get_open_trades()
		assert len(open_trades) > 0
		self.trader.close_trades((base_currency, quote_currency))
		self.assertLess(
			len(self.trader.get_open_trades()),
			len(open_trades)
		)

	def test_close_all_trades(self):

		self.trader.trade(
			TraderTest.TRADES[1].instrument,
			TraderTest.TRADES[1].action,
			TraderTest.TRADES[1].margin
		)
		self.trader.close_all_trades()
		open_trades = self.trader.get_open_trades()
		self.assertEqual(len(open_trades), 0)

	def test_get_candlestick(self):

		candle_sticks: List[CandleStick] = self.trader.get_candlestick(
			instrument=("USD", "CAD"),
			to=datetime.now(),
			granularity="M1",
			count=200
		)

		self.assertEqual(
			len(candle_sticks),
			200
		)

	def test_stop_loss_trade(self):
		current_price = self.trader.get_price(("AUD", "USD"))
		self.trader.trade(
			("AUD", "USD"),
			action=Trader.TraderAction.SELL,
			margin=20,
			stop_loss=0.6566031374999999
		)

		open_trades: typing.List[Trade] = self.trader.get_open_trades()

		self.assertEqual(len(open_trades), 1)
		self.assertIsNotNone(open_trades[0].stopLossOrder)

	def test_get_instrument_precision(self):
		self.assertEqual(
			self.trader.get_instrument_precision(("AUD", "USD")),
			5
		)

if __name__ == "__main__":
	unittest.main()
