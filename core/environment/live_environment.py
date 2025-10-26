import os
import random
from typing import *

import numpy as np
import pandas as pd

from datetime import datetime

from core.di import EnvironmentUtilsProvider, ServiceProvider
from core.utils.research.data.prepare.smoothing_algorithm import SmoothingAlgorithm, KalmanFilter, MovingAverage
from lib.network.oanda import Trader
from lib.network.oanda.data import models
from lib.utils.logger import Logger
from core import Config
from core.environment.trade_state import TradeState, AgentState, MarketState
from core.agent.trader_action import TraderAction
from core.environment.trade_environment import TradeEnvironment


class LiveEnvironment(TradeEnvironment):

	def __init__(
			self,
			*args,
			trader: Optional[Trader] = None,
			instruments: Optional[List[Tuple[str, str]]] = None,
			agent_currency: Tuple[str, str] = Config.AGENT_CURRENCY,
			agent_max_instruments: Optional[int] = Config.AGENT_MAX_INSTRUMENTS,
			agent_use_static_instruments: bool = Config.AGENT_USE_STATIC_INSTRUMENTS,
			market_state_granularity: str = Config.MARKET_STATE_GRANULARITY,
			candlestick_dump_path: str = Config.DUMP_CANDLESTICKS_PATH,
			use_smoothing: int = Config.MARKET_STATE_SMOOTHING,
			**kwargs
	):
		super(LiveEnvironment, self).__init__(*args, **kwargs)
		self.__agent_currency = agent_currency
		self.__agent_max_instruments = agent_max_instruments
		self.__market_state_granularity = market_state_granularity
		self.__trader = trader
		if trader is None:
			self.__trader = EnvironmentUtilsProvider.provide_trader()
		self.__instruments = instruments
		if instruments is None:
			if agent_use_static_instruments:
				self.__instruments = Config.AGENT_STATIC_INSTRUMENTS
			else:
				if agent_max_instruments is not None:
					self.__instruments = self.__get_random_instruments(agent_max_instruments)
				else:
					self.__instruments = self.__trader.get_instruments()
		self.__all_instruments = self.__generate_all_instruments(self.__instruments, self.__agent_currency)
		self.__candlestick_dump_path = candlestick_dump_path
		self.__use_smoothing = use_smoothing
		self.__smoothing_algorithm = None if not use_smoothing else ServiceProvider.provide_smoothing_algorithm()
		Logger.info(f"Using Smoothing Algorithm: {self.__smoothing_algorithm}")

	def __generate_all_instruments(self, instruments, agent_currency) -> List[Tuple[str, str]]:
		valid_instruments = self.__trader.get_instruments()
		currencies = self.__get_currencies(instruments)
		conversion_instruments = []
		for currency in currencies:
			if currency == agent_currency:
				continue
			currency_instrument = (currency, agent_currency)
			if currency_instrument not in valid_instruments:
				currency_instrument = (agent_currency, currency)
			if currency_instrument not in instruments:
				conversion_instruments.append(currency_instrument)

		return instruments + conversion_instruments

	def __get_random_instruments(self, size) -> List[Tuple[str, str]]:
		instruments = self.__trader.get_instruments()
		selected_instruments = None
		while selected_instruments is None or \
			self.__agent_currency not in self.__get_currencies(selected_instruments) or \
			False in [
				(self.__agent_currency, currency) in selected_instruments or
				(currency, self.__agent_currency) in selected_instruments

				for currency in self.__get_currencies(selected_instruments)
				if currency != self.__agent_currency
			] or \
			False in [
				selected_instruments.count(instrument) == 1
				for instrument in selected_instruments
			]:
			selected_instruments = random.choices(instruments, k=size)
		return selected_instruments

	@staticmethod
	def __to_oanda_action(action):
		if action == TraderAction.Action.BUY:
			return Trader.TraderAction.BUY
		if action == TraderAction.Action.SELL:
			return Trader.TraderAction.SELL

	@staticmethod
	def __from_oanda_action(action):
		if action == Trader.TraderAction.BUY:
			return TraderAction.Action.BUY
		if action == Trader.TraderAction.SELL:
			return TraderAction.Action.SELL

	def __get_agent_state(self, market_state: MarketState) -> AgentState:
		balance = self.__trader.get_balance()
		open_trades_raw: List[models.Trade] = self.__trader.get_open_trades()
		open_trades: List[AgentState.OpenTrade] = []
		for trade in open_trades_raw:
			base_currency, quote_currency = trade.get_instrument()
			open_trades.append(
				AgentState.OpenTrade(
					TraderAction(
						base_currency,
						quote_currency,
						self.__from_oanda_action(trade.get_action()),
						margin_used=trade.marginUsed,
						units=trade.get_units()
					),
					trade.price,
					current_value=trade.get_current_price()
				)
			)
		return AgentState(balance, market_state, open_trades=open_trades, margin_rate=self.__trader.get_margin_rate())

	@staticmethod
	def __get_currencies(pairs: List[Tuple[str, str]]) -> List[str]:
		currencies = []
		for pair in pairs:
			currencies += pair
		return list(set(currencies))

	def __get_market_state(self, memory_size, granularity) -> MarketState:
		market_state = MarketState(
			currencies=self.__get_currencies(self.__all_instruments),
			tradable_pairs=self.__instruments,
			memory_len=memory_size
		)

		for base_currency, quote_currency in self.__all_instruments:
			market_state.update_state_of(
				base_currency,
				quote_currency,
				self.__prepare_instrument(base_currency, quote_currency, memory_size, granularity)
			)
			market_state.update_spread_state_of(
				base_currency,
				quote_currency,
				self.__trader.get_spread_price((base_currency, quote_currency)).get_spread_cost()
			)

		return market_state

	def __dump_candlesticks(self, df: pd.DataFrame):
		df.to_csv(
			os.path.join(self.__candlestick_dump_path, f"{datetime.now().timestamp()}.csv")
		)

	@staticmethod
	def __candlesticks_to_dataframe(candlesticks: List[models.CandleStick]) -> pd.DataFrame:
		df_list = []
		for candlestick in candlesticks:
			candle_dict = {key: float(value) for key, value in candlestick.mid.items()}
			candle_dict["v"] = candlestick.volume
			candle_dict["time"] = candlestick.time
			df_list.append(candle_dict)

		return pd.DataFrame(df_list)

	def __process_instrument(self, sequence: np.ndarray) -> np.ndarray:
		if self.__use_smoothing:
			sequence = self.__smoothing_algorithm(sequence)
		return sequence

	def __prepare_instrument(self, base_currency, quote_currency, size, granularity) -> np.ndarray:
		if isinstance(self.__smoothing_algorithm, MovingAverage):
			size = size + self.__smoothing_algorithm.window_size - 1

		candle_sticks = self.__trader.get_candlestick(
			(base_currency, quote_currency),
			count=size,
			to=datetime.now(),
			granularity=granularity
		)
		df = self.__candlesticks_to_dataframe(candle_sticks)
		if self.__candlestick_dump_path is not None:
			self.__dump_candlesticks(df)
		sequence = df["c"].to_numpy()
		sequence = self.__process_instrument(sequence)
		return sequence

	def _open_trade(self, action: TraderAction):
		super()._open_trade(action)
		self.__trader.trade(
			(action.base_currency, action.quote_currency),
			self.__to_oanda_action(action.action),
			action.margin_used,
			time_in_force=Config.DEFAULT_TIME_IN_FORCE
		)

	def _close_trades(self, base_currency, quote_currency):
		super()._close_trades(base_currency, quote_currency)
		self.__trader.close_trades((base_currency, quote_currency))

	def _initiate_state(self) -> TradeState:
		market_state = self.__get_market_state(self._market_state_memory, self.__market_state_granularity)
		agent_state = self.__get_agent_state(market_state)

		return TradeState(
			agent_state=agent_state,
			market_state=market_state
		)

	def _refresh_state(self, state: TradeState = None) -> TradeState:
		if state is None:
			state = self.get_state()
		new_state = self._initiate_state()
		new_state._TradeState__attached_state = state._TradeState__attached_state
		return new_state
