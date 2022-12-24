from typing import *

import math
import datetime
import pytz

from lib.utils.logger import Logger
from .data.models import AccountSummary, Trade, Order, CloseTradeResponse,  CreateOrderResponse, CandleStick, SpreadPrice, \
	ClosedTradeDetails
from . import OandaNetworkClient
from .requests import AccountSummaryRequest, GetOpenTradesRequest, GetInstrumentsRequest, CreateOrderRequest, \
	CloseTradeRequest, GetPriceRequest, GetCandleSticksRequest, GetSpreadPriceRequest, GetClosedTradesRequest
from .exceptions import InstrumentNotFoundException, InvalidActionException, InsufficientMarginException


class Trader:

	INSTRUMENT_DELIMETER = "_"

	class TraderAction:
		SELL = 0
		BUY = 1
		
		@staticmethod
		def reverse(action):
			if action == Trader.TraderAction.BUY:
				return Trader.TraderAction.SELL
			if action == Trader.TraderAction.SELL:
				return Trader.TraderAction.BUY
			raise InvalidActionException()
	
	def __init__(
			self,
			token: str,
			account_no: str,
			timezone: pytz.timezone = None,
			trading_url: str = "https://api-fxpractice.oanda.com/v3",
			timeout: Optional[float] = None
	):
		self.__token: str = token
		self.__account_no: str = account_no
		self.__client = OandaNetworkClient(trading_url, self.__token, self.__account_no, timeout=timeout)
		self.__summary: AccountSummary = self.get_account_summary()
		self.__timezone = timezone
		if timezone is None:
			self.__timezone = pytz.timezone("UTC")
		Logger.info(f"Using timezone {self.__timezone}")

	def get_account_summary(self, update: bool = True) -> AccountSummary:
		summary = self.__client.execute(AccountSummaryRequest())
		if update:
			self.__summary = summary
		return summary

	def get_balance(self) -> float:
		return self.get_account_summary(True).balance

	def get_margin_available(self) -> float:
		return self.get_account_summary(True).marginAvailable

	def get_open_trades(self) -> List[Trade]:
		return self.__client.execute(GetOpenTradesRequest())

	def get_margin_rate(self) -> float:
		return self.get_account_summary().marginRate
	
	def get_instruments(self) -> List[Tuple[str, str]]:
		return self.__client.execute(GetInstrumentsRequest())

	def __get_proper_instrument(self, instrument: Tuple[str, str]) -> Tuple[str, str]:
		available_instruments = self.get_instruments()
		rev_instrument = instrument[::-1]
		for cur_instrument in available_instruments:
			if cur_instrument == instrument:
				return instrument
			if cur_instrument == rev_instrument:
				return rev_instrument
		raise InstrumentNotFoundException()

	def __get_proper_instrument_action_pair(self, instrument, action) -> Tuple[Tuple[str, str], int]:
		proper_instrument = self.__get_proper_instrument(instrument)
		if proper_instrument == instrument:
			return (proper_instrument, action)
		return (proper_instrument, Trader.TraderAction.reverse(action))

	def __get_units(self, action, units) -> int:
		if action == Trader.TraderAction.BUY:
			return units
		elif action == Trader.TraderAction.SELL:
			return -units
		raise InvalidActionException()

	def get_price(self, instrument: Tuple[str, str]) -> float:
		if instrument[0] == instrument[1]:
			return 1
		proper_instrument = self.__get_proper_instrument(instrument)
		price: float = self.__client.execute(
			GetPriceRequest(Trader.format_instrument(proper_instrument))
		).get_price()
		
		if proper_instrument == instrument:
			return price
		return 1/price

	def __localize_datetime(self, dt: datetime) -> datetime:
		return self.__timezone.localize(dt)

	def get_candlestick(self, instrument: Tuple[str, str], from_: datetime = None, to: datetime = None,
							granularity: str = None, count: int = None) -> List[CandleStick]:
		if from_ is not None and from_.tzinfo is None:
			from_ = self.__localize_datetime(from_)
		if to is not None and to.tzinfo is None:
			to = self.__localize_datetime(to)

		return self.__client.execute(
			GetCandleSticksRequest(
				instrument,
				from_=from_,
				to=to,
				granularity=granularity,
				count=count
			)
		)

	def get_spread_price(self, instrument: Tuple[str, str]) -> SpreadPrice:
		return self.__client.execute(
			GetSpreadPriceRequest(
				instrument
			)
		)

	def get_closed_trades(self, count=None) -> List[ClosedTradeDetails]:
		request = GetClosedTradesRequest()
		if count is not None:
			request = GetClosedTradesRequest(count=count)
		return self.__client.execute(
			request
		)

	def __get_margin_required(self, instrument: Tuple[str, str], units: int) -> float:
		in_quote = self.get_price(instrument)*self.__summary.marginRate*units
		quote_price = self.get_price((instrument[1], self.__summary.currency))
		return in_quote * quote_price
	
	def __get_units_for_margin_used(self, instrument: Tuple[str, str], margin_used: float) -> int:
		in_quote = self.get_price((self.__summary.currency, instrument[1])) * margin_used
		price = self.get_price(instrument)
		return math.floor(in_quote / (self.__summary.marginRate * price))

	@Logger.logged_method
	def trade(self, instrument: Tuple, action: int, margin: float, time_in_force="FOK") -> CreateOrderResponse:
		instrument, action = self.__get_proper_instrument_action_pair(instrument, action)
		if self.get_margin_available() < margin:
			raise InsufficientMarginException()
		units = self.__get_units(
			action,
			self.__get_units_for_margin_used(instrument, margin)
		)
		order = Order(units, Trader.format_instrument(instrument), time_in_force)
		return self.__client.execute(
			CreateOrderRequest(order)
		)

	@Logger.logged_method
	def close_trade(self, trade_id: int) -> CloseTradeResponse:
		response: CloseTradeResponse = self.__client.execute(
			CloseTradeRequest(trade_id)
		)
		return response

	@Logger.logged_method
	def close_trades(self, instrument: Tuple[str, str]) -> List[CloseTradeResponse]:
		return [
			self.close_trade(trade.id) 
			for trade in self.get_open_trades() 
			if trade.get_instrument() == instrument or instrument[::-1] == trade.get_instrument()
		]

	@Logger.logged_method
	def close_all_trades(self) -> List[CloseTradeResponse]:
		open_trades = self.get_open_trades()
		return [self.close_trade(trade.id) for trade in open_trades]

	@staticmethod
	def split_instrument(instrument: str) -> Tuple[str, str]:
		return tuple(instrument.split(Trader.INSTRUMENT_DELIMETER))
	
	@staticmethod
	def format_instrument(instrument: Tuple[str, str]) -> str:
		return f"{instrument[0]}{Trader.INSTRUMENT_DELIMETER}{instrument[1]}"
