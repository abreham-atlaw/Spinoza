from typing import *

import attr



@attr.define
class Trade:
	instrument: Tuple[str, str] = attr.ib()


@attr.define
class AccountSummary:

	NAV: float = attr.ib()
	alias: str = attr.ib()
	balance: float = attr.ib()
	currency: str = attr.ib()
	id: str = attr.ib()
	marginAvailable: float = attr.ib()
	marginRate: float = attr.ib()
	marginUsed: float = attr.ib()


@attr.define
class Trade:

	id: str = attr.ib()
	instrument: str = attr.ib()
	initialUnits: int = attr.ib()
	initialMarginRequired: float = attr.ib()
	realizedPL: float = attr.ib()
	unrealizedPL: float = attr.ib()
	marginUsed: float = attr.ib()
	state: str = attr.ib()
	price: float = attr.ib()

	def get_instrument(self) -> Tuple[str, str]:
		from lib.network.oanda import Trader
		return Trader.split_instrument(self.instrument)
	
	def get_action(self) -> int:
		from lib.network.oanda import Trader
		if self.initialUnits < 0:
			return Trader.TraderAction.SELL
		return Trader.TraderAction.BUY
	
	def get_units(self) -> int:
		return abs(self.initialUnits)

	def get_current_price(self) -> float:
		return self.price - (self.unrealizedPL/self.initialUnits)
		#return self.price


@attr.define
class Order:

	units: int = attr.ib()
	instrument: str = attr.ib()
	timeInForce: str = attr.ib()
	type: Optional[str] = "MARKET"
	positionFill: Optional[str] = "DEFAULT"


@attr.define
class TradeOpened:
	units: int = attr.ib()
	tradeID: str = attr.ib()
	initialMarginRequired: float = attr.ib()


@attr.define
class TradeClosed:
	tradeID: int = attr.ib()
	units: int = attr.ib()
	realizedPL: float = attr.ib()


@attr.define
class CreateOrderResponse:

	reason: str = attr.ib()	
	orderID: int = attr.ib()
	requestedUnits: Optional[int] = attr.ib(default=None)
	tradeOpened: Optional[TradeOpened] = attr.ib(default=None)

	def is_successful(self):
		return self.tradeOpened is not None


@attr.define
class CloseTradeResponse:
	orderID: int = attr.ib()
	tradesClosed: Optional[List[TradeClosed]] = attr.ib(default=None)

	def is_successful(self):
		return self.tradesClosed is not None


@attr.define
class Price:
	
	instrument: str = attr.ib()
	closeoutBid: float = attr.ib()
	closeoutAsk: float = attr.ib()

	def get_instrument(self):
		from lib.network.oanda import Trader
		return Trader.split_instrument(self.instrument)

	def get_price(self) -> float:
		return (self.closeoutAsk + self.closeoutBid)/2


@attr.define
class CandleStick:

	volume: int = attr.ib()
	mid: Dict = attr.ib()

	def get_open(self) -> float:
		return self.mid.get("o")

	def get_close(self) -> float:
		return self.mid.get("c")

	def get_high(self) -> float:
		return self.mid.get("h")

	def get_low(self) -> float:
		return self.mid.get("l")


@attr.define
class SpreadPrice:

	closeoutBid: float = attr.ib()
	closeoutAsk: float = attr.ib()

	def get_buy(self) -> float:
		return self.closeoutAsk

	def get_sell(self) -> float:
		return self.closeoutBid

	def get_spread_cost(self) -> float:
		return (self.get_buy() - self.get_sell())/2
