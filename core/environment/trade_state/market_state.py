import typing
from typing import *

import numpy as np


from lib.utils.logger import Logger
from core import Config
from .exceptions import CurrencyNotFoundException


class MarketState:

	def __init__(
			self,
			currencies=None,
			state=None,
			spread_state=None,
			memory_len=None,
			tradable_pairs=None,
			anchor: 'MarketState' = None,
			channels: int = 1,
			close_channel: int = 0
	):
		self.__anchor = anchor

		self.__currencies = currencies
		if currencies is None:
			self.__currencies = Config.CURRENCIES

		if state is None and anchor is None and memory_len is None:
			raise Exception("Insufficient information given on State.")

		if state is None:
			self.__state = self.__init_state(len(currencies), memory_len, channels=channels)
		else:
			self.__state = state

		self.__tradable_pairs: List[Tuple[str, str]] = tradable_pairs
		if tradable_pairs is None:
			self.__tradable_pairs = [
				(base_currency, quote_currency)
				for base_currency in self.__currencies
				for quote_currency in self.__currencies
				if base_currency != quote_currency
			]

		self.__spread_state = spread_state
		if spread_state is None:
			self.__spread_state = np.zeros((len(currencies), len(currencies))).astype('float64')

		self.__close_channel = close_channel

	@property
	def is_anchored(self) -> bool:
		return self.__anchor is not None

	@property
	def channels(self) -> int:
		return self.__state.shape[0]

	def __init_state(self, num_currencies, memory_len, channels: int) -> np.ndarray:

		if self.is_anchored:
			state = np.zeros((channels, num_currencies, num_currencies, 0)).astype('float64')
		else:
			state = np.zeros((channels, num_currencies, num_currencies, memory_len)).astype('float64')

		for i in range(num_currencies):
			state[:, i, i] = 1

		return state

	def __get_currencies_position(self, base_currency, quote_currency):
		if base_currency not in self.__currencies:
			raise CurrencyNotFoundException(base_currency)
		if quote_currency not in self.__currencies:
			raise CurrencyNotFoundException(quote_currency)

		return self.__currencies.index(base_currency), self.__currencies.index(quote_currency)

	def get_state_of(self, base_currency, quote_currency) -> np.ndarray:
		return np.squeeze(
			self.get_channels_state(base_currency, quote_currency, channels=[self.__close_channel]),
			axis=0
		)

	def get_channels_state(self, base_currency, quote_currency, channels: typing.List[int] = None) -> np.ndarray:
		if channels is None:
			channels = np.arange(self.__state.shape[0])
		bci, qci = self.__get_currencies_position(base_currency, quote_currency)
		state = self.__state[channels, bci, qci]
		if self.is_anchored:
			state = np.concatenate(
				(self.__anchor.get_channels_state(base_currency, quote_currency, channels=channels), state)
			)[:, state.shape[0]:]
		return state

	def get_current_price(self, base_currency, quote_currency) -> np.float32:
		"""
		TODO: OPTIMIZE. If self.is_anchored and self.__state.shape[-1] > 0 return self.__state[bci, qci, -1]
		TODO: (no need to acquire and concatenate anchor.get_state_of(...)
		"""
		return self.get_state_of(base_currency, quote_currency)[-1]

	def get_spread_state_of(self, base_currency, quote_currency) -> float:
		bci, qci = self.__get_currencies_position(base_currency, quote_currency)
		return self.__spread_state[bci, qci]

	def __add_empty_state_layer(self, size: int):
		self.__state = np.concatenate((self.__state, np.zeros(self.__state.shape[:-1] + [size])), axis=-1)

	def update_state_of(self, base_currency, quote_currency, values: np.ndarray):
		bci, qci = self.__get_currencies_position(base_currency, quote_currency)

		if self.is_anchored:
			self.__add_empty_state_layer(len(values))
			self.__state[:, bci, qci, -len(values):] = values
		else:
			self.__state[:, bci, qci] = np.concatenate((self.__state[:, bci, qci, values.shape[-1]:], values), axis=-1)

		self.__state[:, qci, bci] = 1/self.__state[:, bci, qci]

	def update_state_layer(self, state_layer: np.ndarray):
		for i in range(state_layer.shape[0]):
			for j in range(state_layer.shape[1]):
				if not np.isclose(state_layer[i, j], 1/state_layer[j, i]):
					Logger.warning(f"Inconsistent Layer given. {state_layer[i, j], state_layer[j, i]}")
		if self.is_anchored:
			self.__add_empty_state_layer(1)
		else:
			self.__state[...,:-1] = self.__state[..., 1:]

		self.__state[..., -1] = state_layer

		for i in range(len(self.__currencies)):
			self.__state[:, i, i, -1] = 1

	def update_spread_state_of(self, base_currency, quote_currency, value: float):
		bci, qci = self.__get_currencies_position(base_currency, quote_currency)
		self.__spread_state[bci, qci] = value
		self.__spread_state[qci, bci] = self.convert(value, to=base_currency, from_=quote_currency)

	def set_spread_state(self, spread_state: np.ndarray):
		self.__spread_state = spread_state

	def get_price_matrix(self) -> np.ndarray:
		return self.__state  # TODO: INCLUDE ANCHORED STATE AS WELL

	def get_spread_matrix(self) -> np.ndarray:
		return self.__spread_state

	def get_memory_len(self) -> int:
		if self.is_anchored:
			return self.__anchor.get_price_matrix().shape[-1]
		return self.__state.shape[-1]

	def get_currencies(self) -> List[str]:
		return self.__currencies

	def get_tradable_pairs(self) -> List[Tuple[str, str]]:
		return self.__tradable_pairs

	def convert(self, value, from_, to):
		return value * self.get_current_price(from_, to)

	def __deepcopy__(self, memo=None):
		if Config.MARKET_STATE_USE_ANCHOR:
			state, anchor = None, self
		else:
			state, anchor = self.__state.copy(), None
		return MarketState(
			currencies=self.__currencies.copy(),
			tradable_pairs=self.__tradable_pairs.copy(),
			state=state,
			spread_state=self.__spread_state.copy(),
			anchor=anchor
		)

	def __hash__(self):
		return hash((tuple(self.__currencies), tuple(self.__tradable_pairs), self.__state.tobytes(), self.__spread_state.tobytes()))

	def __eq__(self, other):
		if not isinstance(other, MarketState):
			return False
		return \
			(self.__currencies == other._MarketState__currencies) and \
			(self.__tradable_pairs == other._MarketState__tradable_pairs) and \
			(np.all(self.__state == other._MarketState__state)) and \
			(np.all(self.__spread_state == other._MarketState__spread_state))