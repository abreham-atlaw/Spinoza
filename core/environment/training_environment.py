from typing import *

import numpy as np

import datetime
from copy import deepcopy

from core.environment.trade_state import TradeState, AgentState, MarketState
from core.agent.action import TraderAction
from core import Config
from .trade_environment import TradeEnvironment


class TrainingEnvironment(TradeEnvironment):

	def __init__(self, initial_balance=100, pg_config=None, historical_table_name=Config.HISTORICAL_TABLE_NAME,
				currencies_table_name=Config.CURRENCIES_TABLE_NAME,
				tradeable_pairs_table_name=Config.TRADEABLE_PAIRS_TABLE_NAME,
				datetimes_table_name=Config.DISTINCT_DATETIMES_TABLE_NAME,
				datetime_format="%Y-%m-%d %H:%M:*"
				):

		super(TrainingEnvironment, self).__init__()
		self.__initial_balance = initial_balance
		self.__pg_config = pg_config
		if pg_config is None:
			self.__pg_config = Config.DEFAULT_PG_CONFIG
		self.__historical_table_name = historical_table_name
		self.__currencies_table_name = currencies_table_name
		self.__tradeable_pairs_table_name = tradeable_pairs_table_name
		self.__distinct_datetime_table_name = datetimes_table_name
		self.datetime_format = datetime_format
		self.last_datetime = None

	def _initiate_state(self) -> TradeState:
		market_state = self.__initiate_market_state(Config.MARKET_STATE_MEMORY)
		agent_state = AgentState(self.__initial_balance, market_state)
		return TradeState(agent_state=agent_state, market_state=market_state)

	def __prepare_timestep_state_layer(self, currencies: List[str], datetime_: datetime.datetime) -> np.ndarray:
		state_layer = np.zeros((len(currencies), len(currencies)))

		cursor = pg_connection.cursor()

		cursor.execute(
			f"SELECT base_currency, quote_currency, close FROM {self.__historical_table_name} WHERE datetime=%s",
			(datetime_.strftime(self.datetime_format),)
		)

		for base_currency, quote_currency, value in cursor.fetchall():

			if base_currency not in currencies or quote_currency not in currencies:
				continue

			state_layer[
				currencies.index(base_currency),
				currencies.index(quote_currency)
			] = value

			state_layer[
				currencies.index(quote_currency),
				currencies.index(base_currency)
			] = 1/value

		for i in range(len(currencies)):
			state_layer[i][i] = 1

		cursor.close()

		return state_layer

	def __initiate_market_state(self, memory_size) -> MarketState:

		cursor = pg_connection.cursor()

		cursor.execute(f"SELECT currency FROM {self.__currencies_table_name}")
		currencies = [currency[0] for currency in cursor.fetchall()]

		cursor.execute(f"SELECT base_currency, quote_currency FROM {self.__tradeable_pairs_table_name}")
		tradeable_pairs = cursor.fetchall()

		market_state = MarketState(currencies=currencies, memory_len=memory_size, tradable_pairs=tradeable_pairs)

		cursor.execute(
			f"SELECT datetime FROM {self.__distinct_datetime_table_name} ORDER BY datetime ASC LIMIT {memory_size};"
		)

		datetimes: List[Tuple[datetime.datetime]] = cursor.fetchall()

		for i, (datetime_,) in enumerate(datetimes):
			market_state.update_state_layer(
				self.__prepare_timestep_state_layer(currencies, datetime_)
			)

		self.last_datetime = datetimes[-1][0]

		cursor.close()

		return market_state

	def _refresh_state(self, state=None) -> TradeState:
		if state is None:
			state = self.get_state()
		state = deepcopy(state)
		cursor = pg_connection.cursor()
		cursor.execute(
			f"SELECT DISTINCT datetime FROM {self.__distinct_datetime_table_name} "
			f"WHERE datetime > %s ORDER BY datetime ASC LIMIT 1;",
			(self.last_datetime.strftime(self.datetime_format), )
		)
		(next_datetime_,) = cursor.fetchone()
		state.get_market_state().update_state_layer(
			self.__prepare_timestep_state_layer(
				self._state.market_state.get_currencies(),
				next_datetime_
			)
		)
		return state

	def get_valid_actions(self, state=None) -> List[Union[TraderAction, None]]:
		actions = super().get_valid_actions(state)
		if state is None:
			state = self.get_state()

		return [
			action
			for action in actions
			if(
					action is None or
					(
						0 not in state.get_market_state().get_state_of(action.base_currency, action.quote_currency)
					)
			)
		]
