import typing
from abc import ABC, abstractmethod
from copy import deepcopy

import numpy as np

from core.agent.action import TraderAction, Action, ActionSequence
from core.environment.trade_state import TradeState, AgentState
from lib.rl.agent import ActionChoiceAgent
from core import Config
from lib.utils.logger import Logger


class ActionChoiceTrader(ActionChoiceAgent, ABC):

	def __init__(
			self,
			*args,
			trade_size_gap=Config.AGENT_TRADE_SIZE_GAP,
			trade_size_use_percentage=Config.AGENT_TRADE_SIZE_USE_PERCENTAGE,
			trade_min_size=Config.AGENT_TRADE_MIN_SIZE,
			multi_actions=Config.AGENT_SUPPORT_MULTI_ACTION,
			stop_loss_granularity=Config.AGENT_STOP_LOSS_GRANULARITY,
			stop_loss_value_bound=Config.AGENT_STOP_LOSS_VALUE_BOUND,
			**kwargs
	):
		super().__init__(*args, **kwargs)
		self.__trade_size_gap = trade_size_gap
		self.__trade_size_use_percentage = trade_size_use_percentage
		self.__trade_min_size = trade_min_size
		self.__multi_actions = multi_actions
		self.__stop_loss_granularity = stop_loss_granularity
		self.__stop_loss_value_bound = stop_loss_value_bound

		Logger.info(f"[ActionChoiceTrader]: Multi Action Support={multi_actions}")

	@abstractmethod
	def _simulate_action(self, state: TradeState, action: Action):
		pass

	def _generate_stop_loss_bounds(self, action: int) -> typing.List[float]:
		direction = -1 if action == TraderAction.Action.SELL else 1
		return [
			1 + (-direction)*r
			for r in np.arange(self.__stop_loss_value_bound[0], self.__stop_loss_value_bound[1], self.__stop_loss_granularity)
		]

	def _generate_lone_actions(self, state: TradeState) -> typing.List[TraderAction]:
		pairs = state.get_market_state().get_tradable_pairs()

		gap = self.__trade_size_gap * state.get_agent_state().get_margin_available() if self.__trade_size_use_percentage \
			else self.__trade_size_gap
		min_size = self.__trade_min_size * state.get_agent_state().get_balance(original=True) if self.__trade_size_use_percentage \
			else self.__trade_min_size

		amounts = list(filter(
			lambda size: size >= min_size,
			[
				(i + 1) * gap
				for i in range(int(state.get_agent_state().get_margin_available() // gap))
			]
		))

		actions: typing.List[typing.Optional[TraderAction]] = [
			TraderAction(
				pair[0],
				pair[1],
				action,
				margin_used=amount,
				stop_loss=stop_loss
			)
			for pair in pairs
			for action in [TraderAction.Action.BUY, TraderAction.Action.SELL]
			for stop_loss in self._generate_stop_loss_bounds(action)
			for amount in amounts
		]

		actions += [
			TraderAction(trade.get_trade().base_currency, trade.get_trade().quote_currency, TraderAction.Action.CLOSE)
			for trade in state.get_agent_state().get_open_trades()
		]

		return actions

	def __generate_reversal_actions(self, trade: AgentState.OpenTrade, state: TradeState) -> typing.List[ActionSequence]:
		state = deepcopy(state)

		close_action = TraderAction(
			trade.get_trade().base_currency,
			trade.get_trade().quote_currency,
			TraderAction.Action.CLOSE
		)

		self._simulate_action(state, close_action)

		return [
			ActionSequence(
				actions=(
					close_action,
					action
				)
			)
			for action in self._generate_lone_actions(state)
			if action.action not in [
				trade.get_trade().action,
				TraderAction.Action.CLOSE
			]
		]

	def __generate_action_sequences(self, state: TradeState) -> typing.List[ActionSequence]:
		if len(state.get_agent_state().get_open_trades()) == 0:
			return []

		actions = []
		for trade in state.get_agent_state().get_open_trades():
			actions.extend(self.__generate_reversal_actions(trade, state))

		return actions

	def _generate_actions(self, state: TradeState) -> typing.List[typing.Optional[Action]]:
		actions = self._generate_lone_actions(state)

		actions.append(None)

		if self.__multi_actions:
			actions.extend(self.__generate_action_sequences(state))

		return actions


