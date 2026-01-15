import os
import random
import typing
from typing import *

import numpy as np
import pandas as pd

from datetime import datetime

from core.di import EnvironmentUtilsProvider, ServiceProvider
from core.utils.research.data.prepare.smoothing_algorithm import MovingAverage
from core.utils.research.data.prepare.utils.data_prep_utils import DataPrepUtils
from lib.network.oanda import Trader
from lib.network.oanda.data import models
from lib.network.oanda.exceptions import InsufficientMarginException
from lib.utils.cache.decorators import CacheDecorators
from lib.utils.logger import Logger
from core import Config
from core.environment.trade_state import TradeState, AgentState, MarketState
from core.agent.action import TraderAction
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
			channels: typing.Tuple[str,...] = Config.MARKET_STATE_CHANNELS,
			smoothed_channels: typing.Tuple[str,...] = Config.MARKET_STATE_SMOOTHED_CHANNELS,
			stop_loss_conversion: bool = Config.AGENT_STOP_LOSS_CONVERSION,
			stop_loss_conversion_bounds: typing.Tuple[float, float] = Config.AGENT_STOP_LOSS_CONVERSION_BOUNDS,
			stop_loss_conversion_accuracy: int = Config.AGENT_STOP_LOSS_CONVERSION_ACCURACY,
			trigger_value_absolute_multiplier: float = Config.AGENT_TRIGGER_ABSOLUTE_VALUE_MULTIPLIER,
			close_channel_label: str = "c",
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
		self.__channels = channels
		self.__close_channel_label = close_channel_label
		self.__smoothed_channels = smoothed_channels
		self.__smoothing_algorithm = None if not use_smoothing else ServiceProvider.provide_smoothing_algorithm()

		self.__trigger_value_conversion = stop_loss_conversion
		self.__stop_loss_conversion_bounds = stop_loss_conversion_bounds
		self.__stop_loss_conversion_accuracy = stop_loss_conversion_accuracy
		self.__trigger_value_absolute_multiplier = trigger_value_absolute_multiplier
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
			memory_len=memory_size,
			channels=len(self.__channels),
			close_channel=self.__channels.index(self.__close_channel_label)
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

		df = pd.DataFrame(df_list)
		df = DataPrepUtils.encode_timestamp(df)
		return df

	def __process_instrument(self, data: np.ndarray) -> np.ndarray:
		if not self.__use_smoothing:
			return data

		data = np.stack([
			self.__smoothing_algorithm(data[i, :])
			if self.__channels[i] in self.__smoothed_channels
			else data[i, self.__smoothing_algorithm.reduction:]
			for i in range(data.shape[0])
		])

		return data

	CacheDecorators.cached_method(timeout=Config.ENVIRONMENT_FETCH_CACHE_TIMEOUT)
	def __fetch_instrument_state(self, base_currency, quote_currency, size, granularity) -> np.ndarray:
		size = size + self.__smoothing_algorithm.reduction

		candle_sticks = self.__trader.get_candlestick(
			(base_currency, quote_currency),
			count=size,
			to=datetime.now(),
			granularity=granularity
		)
		df = self.__candlesticks_to_dataframe(candle_sticks)
		if self.__candlestick_dump_path is not None:
			self.__dump_candlesticks(df)

		data = df[list(self.__channels)].to_numpy().transpose()
		return data

	def __prepare_instrument(self, base_currency, quote_currency, size, granularity) -> np.ndarray:
		data = self.__fetch_instrument_state(base_currency, quote_currency, size, granularity)
		data = self.__process_instrument(data)
		return data

	def __convert_stop_loss(self, action: TraderAction):
		Logger.info(f"Converting Stop Loss...")
		p = np.linspace(*self.__stop_loss_conversion_bounds, self.__stop_loss_conversion_accuracy)

		x = self.__fetch_instrument_state(action.base_currency, action.quote_currency, self._market_state_memory, self.__market_state_granularity)
		if x.ndim == 2:
			x = x[0]

		y = np.repeat(np.expand_dims(x, axis=0), repeats=p.shape[0], axis=0)
		y[:, :-1] = y[:, 1:]
		y[:, -1] *= p
		y = self.__smoothing_algorithm.apply_on_batch(y)
		y = y[:, -1] / y[:, -2]
		y = p[np.argmin(np.abs(y - action.stop_loss))]
		return y

	def __apply_trigger_value_multiplier(self, value: float) -> float:
		new_value = 1 + (value  - 1)*self.__trigger_value_absolute_multiplier
		Logger.info(f"Trigger Value Multiplied: {value} -> {new_value}")
		return new_value

	def __calculate_stop_loss(self, action: TraderAction, trigger_value: float) -> typing.Optional[float]:
		trigger_value = self.__apply_trigger_value_multiplier(trigger_value)

		if self.__trigger_value_conversion:
			trigger_value = self.__convert_stop_loss(action)


		price = self.__trader.get_price((action.base_currency, action.quote_currency))
		return price * trigger_value

	def _open_trade(self, action: TraderAction):
		super()._open_trade(action)
		try:
			self.__trader.trade(
				(action.base_currency, action.quote_currency),
				self.__to_oanda_action(action.action),
				action.margin_used,
				time_in_force=Config.DEFAULT_TIME_IN_FORCE,
				stop_loss=None if action.stop_loss is None else self.__calculate_stop_loss(action, action.stop_loss),
				take_profit=None if action.take_profit is None else self.__calculate_stop_loss(action, action.take_profit)
			)
		except InsufficientMarginException as ex:
			Logger.error(f"Insufficient margin requested for trade: {str(ex)}")
			return

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
