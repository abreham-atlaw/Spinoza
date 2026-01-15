import copy
from typing import *

import math

from core.agent.action import TraderAction, ActionSequence
from core import Config
from .market_state import MarketState
from .exceptions import InsufficientFundsException


class AgentState:
	class OpenTrade:

		def __init__(self, trade: TraderAction, enter_value, current_value=None):
			self.__trade = trade
			self.__enter_value = enter_value
			self.__current_value = current_value
			if current_value is None:
				self.__current_value = enter_value

		def update_current_value(self, value):
			self.__current_value = value

		def get_unrealized_profit(self) -> float:  # RETURNS PROFIT IN TERMS OF QUOTE CURRENCY
			profit = (self.__current_value - self.__enter_value) * self.__trade.units

			if self.__trade.action == TraderAction.Action.SELL:
				profit *= -1

			return profit

		def get_return(self) -> float:
			return self.__trade.margin_used + self.get_unrealized_profit()

		def get_trade(self) -> TraderAction:
			return self.__trade

		def get_enter_value(self) -> float:
			return self.__enter_value

		def get_current_value(self) -> float:
			return self.__current_value

		def __deepcopy__(self, memo=None):
			return AgentState.OpenTrade(
				trade=self.__trade.__deepcopy__(),
				enter_value=self.__enter_value,
				current_value=self.__current_value
			)

		def __hash__(self):
			return hash((self.__trade, self.__enter_value, self.__current_value))

		def __eq__(self, other):
			return isinstance(other, AgentState.OpenTrade) and \
				self.get_current_value() == other.get_current_value() and \
				self.get_enter_value() == other.get_enter_value() and \
				self.get_trade() == other.get_trade()

	def __init__(
			self,
			balance,
			market_state: MarketState,
			margin_rate=1,
			currency=Config.AGENT_CURRENCY,
			open_trades=None,
			core_pricing=Config.AGENT_CORE_PRICING,
			commission_cost=Config.AGENT_COMMISSION_COST,
			trade_penalty=Config.AGENT_TRADE_PENALTY
	):
		self.__balance = balance
		self.__market_state = market_state
		self.__currency = currency
		self.__margin_rate = margin_rate
		self.__core_pricing = core_pricing
		self.__commission_cost = commission_cost
		self.__trade_penalty = trade_penalty
		self.__open_trades: List[AgentState.OpenTrade] = open_trades
		if open_trades is None:
			self.__open_trades: List[AgentState.OpenTrade] = []

	def __update_open_trades(self):
		for trade in self.__open_trades:
			trade.update_current_value(
				self.__market_state.get_current_price(
					trade.get_trade().base_currency,
					trade.get_trade().quote_currency
				)
			)

	def calc_required_margin(self, units: int, base_currency: str, quote_currency: str) -> float:
		price = self.__market_state.get_current_price(base_currency, quote_currency)
		in_quote = price * self.__margin_rate * units
		return in_quote * self.__market_state.get_current_price(quote_currency, self.__currency)

	def __units_for(self, margin: float, base_currency: str, quote_currency: str) -> int:
		price = self.__market_state.get_current_price(base_currency, quote_currency)
		in_quote = margin * self.__market_state.get_current_price(self.__currency, quote_currency)
		return math.floor(in_quote / (self.__margin_rate * price + Config.DEFAULT_EPSILON))

	def get_balance(self, original=False):
		if original:
			return self.__balance
		self.__update_open_trades()
		return self.__balance + sum([
			self.to_agent_currency(
				value=trade.get_unrealized_profit(),
				from_currency=trade.get_trade().quote_currency
			)
			for trade in self.__open_trades
		])

	def set_balance(self, balance):
		self.__balance = balance

	def update_balance(self, delta):
		self.__balance += delta

	def get_margin_used(self) -> float:
		return sum([trade.get_trade().margin_used for trade in self.__open_trades])

	def get_margin_available(self):
		return self.get_balance(original=True) - self.get_margin_used()

	def get_open_trades(self, base_currency=None, quote_currency=None) -> List[OpenTrade]:

		return [
			trade
			for trade in self.__open_trades
			if (trade.get_trade().base_currency == base_currency or base_currency is None) and
			(trade.get_trade().quote_currency == quote_currency or quote_currency is None)
		]

	def rectify_action(self, action: TraderAction):
		if action is None:
			return action
		if isinstance(action, ActionSequence):
			[
				self.rectify_action(a)
				for a in action.actions
			]
			return action

		if action.action == TraderAction.Action.CLOSE:
			return action

		if action.margin_used is None:
			action.margin_used = self.calc_required_margin(action.units, action.base_currency, action.quote_currency)
		elif action.units is None:
			action.units = self.__units_for(action.margin_used, action.base_currency, action.quote_currency)

		return action

	def open_trade(self, action: TraderAction, current_value: float = None):
		if current_value is None:
			current_value = self.__market_state.get_current_price(action.base_currency, action.quote_currency)

		if action.margin_used > self.get_margin_available():
			raise InsufficientFundsException(
				available_margin=self.get_margin_available(),
				requested_margin=action.margin_used
			)

		enter_value = current_value + ((action.action - 0.5) * 2) * self.__market_state.get_spread_state_of(
			action.base_currency, action.quote_currency)

		# self.__balance -= action.units * self.__to_agent_currency(
		# 	value=self.__market_state.get_spread_state_of(action.base_currency, action.quote_currency),
		# 	from_currency=action.quote_currency
		# )

		if self.__core_pricing:
			self.__balance -= self.__commission_cost

		self.__balance -= self.__trade_penalty

		self.__open_trades.append(
			AgentState.OpenTrade(action, enter_value)
		)

	def to_agent_currency(self, value, from_currency) -> float:
		return self.__market_state.convert(value, to=self.__currency, from_=from_currency)

	def add_open_trade(self, trade: OpenTrade):
		self.__open_trades.append(trade)

	def close_trades(self, base_currency, quote_currency, modify_balance=True, close_price: float = None):
		self.__update_open_trades()
		open_trades = self.get_open_trades(base_currency, quote_currency)

		if close_price is not None:
			for trade in open_trades:
				trade.update_current_value(close_price)

		if modify_balance:
			self.update_balance(
				sum([
					self.to_agent_currency(
						value=trade.get_unrealized_profit(),
						from_currency=trade.get_trade().quote_currency
					)
					for trade in open_trades
				])
			)
		self.__open_trades = [trade for trade in self.__open_trades if trade not in open_trades]

	def __deepcopy__(self, memo=None):
		if memo is None:
			memo = {}

		market_state = memo.get('market_state')
		if market_state is None:
			market_state = copy.deepcopy(self.__market_state)

		open_trades = [trade.__deepcopy__() for trade in self.__open_trades]

		return AgentState(
			self.__balance,
			market_state,
			margin_rate=self.__margin_rate,
			currency=self.__currency,
			open_trades=open_trades
		)

	def __hash__(self):
		return hash((self.__balance, tuple(self.__open_trades)))
