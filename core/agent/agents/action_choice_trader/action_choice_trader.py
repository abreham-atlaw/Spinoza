import typing
from abc import ABC

from core.agent.action import TraderAction, Action, ActionSequence
from core.environment.trade_state import TradeState
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
			**kwargs
	):
		super().__init__(*args, **kwargs)
		self.__trade_size_gap = trade_size_gap
		self.__trade_size_use_percentage = trade_size_use_percentage
		self.__trade_min_size = trade_min_size
		self.__multi_actions = multi_actions

		Logger.info(f"[ActionChoiceTrader]: Multi Action Support={multi_actions}")

	@staticmethod
	def __generate_action_sequences(state: TradeState) -> typing.List[ActionSequence]:
		if len(state.get_agent_state().get_open_trades()) == 0:
			return []

		return [
			ActionSequence(
				actions=tuple([
					TraderAction(
						trade.get_trade().base_currency,
						trade.get_trade().quote_currency,
						TraderAction.Action.CLOSE,
					),
					TraderAction(
						trade.get_trade().base_currency,
						trade.get_trade().quote_currency,
						TraderAction.Action.BUY if trade.get_trade().action == TraderAction.Action.SELL else TraderAction.Action.SELL,
						margin_used=trade.get_trade().margin_used
					)
				])
			)
			for trade in state.get_agent_state().get_open_trades()
		]

	def _generate_actions(self, state: TradeState) -> typing.List[typing.Optional[Action]]:
		pairs = state.get_market_state().get_tradable_pairs()

		gap = self.__trade_size_gap * state.get_agent_state().get_margin_available() if self.__trade_size_use_percentage \
			else self.__trade_size_gap
		min_size = self.__trade_min_size * state.get_agent_state().get_balance() if self.__trade_size_use_percentage \
			else self.__trade_min_size

		amounts = list(filter(
			lambda size: size >= min_size,
			[
				(i + 1) * gap
				for i in range(int(state.get_agent_state().get_margin_available() // gap))
			]
		))

		actions: typing.List[typing.Optional[Action]] = [
			TraderAction(
				pair[0],
				pair[1],
				action,
				margin_used=amount
			)
			for pair in pairs
			for action in [TraderAction.Action.BUY, TraderAction.Action.SELL]
			for amount in amounts
		]

		actions += [
			TraderAction(trade.get_trade().base_currency, trade.get_trade().quote_currency, TraderAction.Action.CLOSE)
			for trade in state.get_agent_state().get_open_trades()
		]

		actions.append(None)

		if self.__multi_actions:
			actions.extend(self.__generate_action_sequences(state))

		return actions


