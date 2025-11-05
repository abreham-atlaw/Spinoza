import typing
from abc import ABC

import numpy as np
from tensorflow.keras import models

from lib.rl.agent import ActionRecommendationAgent
from lib.utils.logger import Logger
from core.agent.action import TraderAction
from core.environment.trade_state import TradeState, AgentState
from core import Config
from core.agent.utils.dnn_models import KerasModelHandler


class ActionRecommendationTrader(ActionRecommendationAgent, ABC):

	def __init__(
			self,
			*args,
			num_open_trades: int = Config.AGENT_MAX_OPEN_TRADES,
			ara_model_path: str = Config.ARA_MODEL_CONFIG.path,
			**kwargs
	):
		super().__init__(*args, **kwargs)
		self.__num_open_trades = num_open_trades
		self.__model_path = ara_model_path

	def _init_ara_model(self) -> models.Model:
		Logger.info("Loading Action Recommendation Model...")
		return KerasModelHandler.load_model(self.__model_path)

	@staticmethod
	def __serialize_open_trade(state: TradeState, trade: AgentState.OpenTrade) -> np.ndarray:
		value = [
			trade.get_enter_value(),
			int(trade.get_trade().action == TraderAction.Action.BUY),
			int(trade.get_trade().action == TraderAction.Action.SELL),
			trade.get_trade().margin_used,
		]
		value += state.get_market_state().get_state_of(trade.get_trade().base_currency, trade.get_trade().quote_currency)
		return np.array(value)

	def __serialize_open_trades(self, state: TradeState) -> np.ndarray:
		trades = np.zeros((self.__num_open_trades, 5+state.get_market_state().get_memory_len()))
		for i, trade in enumerate(state.get_agent_state().get_open_trades()):
			trades[i, 1:] = self.__serialize_open_trade(state, trade)
			trades[i, 0] = 1
		return trades

	def _prepare_input(self, state: TradeState, index: int) -> np.ndarray:
		return np.expand_dims(np.concatenate([
			state.get_market_state().get_price_matrix().flatten(),
			state.get_market_state().get_spread_matrix().flatten(),
			self.__serialize_open_trades(state).flatten(),
			[state.get_agent_state().get_balance(), state.get_agent_state().get_margin_available()]
		]), 0)

	@staticmethod
	def __get_class(classes: typing.List[object], values: typing.List[float]) -> typing.Any:
		return max(classes, key=lambda class_: values[classes.index(class_)])

	@staticmethod
	def __one_hot_encoding(classes: typing.List[typing.Any], class_: typing.Any) -> typing.List[float]:
		return [1 if class_ == c else 0 for c in classes]

	def _prepare_output(self, state: TradeState, output: np.ndarray) -> TraderAction:
		output = output.flatten()

		instruments = state.get_market_state().get_tradable_pairs()
		instrument = self.__get_class(
			instruments,
			output[:len(instruments)]
		)

		action = self.__get_class(
			[TraderAction.Action.BUY, TraderAction.Action.SELL, TraderAction.Action.CLOSE],
			output[len(instruments):len(instruments)+2]
		)
		margin = output[-1]
		return TraderAction(
			base_currency=instrument[0],
			quote_currency=instrument[1],
			action=action,
			margin_used=margin
		)

	def _prepare_train_output(self, state: TradeState, action: TraderAction) -> np.ndarray:
		output = []
		output.extend(
			self.__one_hot_encoding(
				state.get_market_state().get_tradable_pairs(),
				(action.base_currency, action.quote_currency)
			)
		)
		output.extend(
			self.__one_hot_encoding(
				[TraderAction.Action.BUY, TraderAction.Action.SELL, TraderAction.Action.CLOSE],
				action.action
			)
		)
		output.append(action.margin_used)
		return np.array(output)
