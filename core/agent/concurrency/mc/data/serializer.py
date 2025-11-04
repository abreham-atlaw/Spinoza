from typing import *

from lib.concurrency.mc.data.serializers import NodeSerializer
from lib.network.rest_interface import Serializer, NumpySerializer
from core.agent.action import TraderAction, Action, ActionSequence
from core.environment.trade_state import TradeState, MarketState, AgentState


class TraderActionSerializer(Serializer):

	def __init__(self):
		super().__init__(TraderAction)

	def serialize(self, data: object) -> Dict:
		if data is None:
			return None
		return data.__dict__.copy()

	def deserialize(self, json_: Dict) -> object:
		if json_ is None:
			return None
		action = TraderAction(None, None, None, None, None)
		action.__dict__ = json_.copy()
		return action


class ActionSequenceSerializer(Serializer):

	def __init__(self):
		super().__init__(ActionSequence)
		self.__trader_action_serializer = TraderActionSerializer()

	def serialize(self, data: ActionSequence) -> Dict:
		return {
			"actions": self.__trader_action_serializer.serialize_many(data.actions)
		}

	def deserialize(self, json_: Dict) -> object:
		return ActionSequence(
			actions=self.__trader_action_serializer.deserialize_many(json_["actions"])
		)


class ActionSerializer(Serializer):

	def __init__(self):
		super().__init__(Action)
		self.__trader_action_serializer = TraderActionSerializer()
		self.__action_sequence_serializer = ActionSequenceSerializer()


	def serialize(self, data: object) -> Dict:
		if isinstance(data, ActionSequence):
			return self.__action_sequence_serializer.serialize(data)
		return self.__trader_action_serializer.serialize(data)

	def deserialize(self, json_: Dict) -> object:
		if json_ is None:
			return None
		if "actions" in json_:
			return self.__action_sequence_serializer.deserialize(json_)
		return self.__trader_action_serializer.deserialize(json_)


class TraderNodeSerializer(NodeSerializer):

	def _init_action_serializer(self) -> Serializer:
		return ActionSerializer()


class MarketStateSerializer(Serializer):

	def __init__(self):
		super().__init__(MarketState)
		self.__numpy_serializer = NumpySerializer()

	def serialize(self, state: MarketState) -> Dict:
		return {
			"currencies": state._MarketState__currencies.copy(),
			"state": self.__numpy_serializer.serialize(
				state._MarketState__state
			),
			"tradable_pairs": state._MarketState__tradable_pairs,
			"spread_state": self.__numpy_serializer.serialize(
				state._MarketState__spread_state
			),
		}

	def deserialize(self, json_: Dict) -> MarketState:
		return MarketState(
			currencies=json_["currencies"],
			tradable_pairs=json_["tradable_pairs"],
			state=self.__numpy_serializer.deserialize(
				json_["state"]
			),
			spread_state=self.__numpy_serializer.deserialize(
				json_["spread_state"]
			)
		)


class OpenTradeSerializer(Serializer):

	def __init__(self):
		super().__init__(AgentState.OpenTrade)
		self.__action_serializer = TraderActionSerializer()

	def serialize(self, trade: AgentState.OpenTrade) -> Dict:
		return {
			"trade": self.__action_serializer.serialize(trade.get_trade()),
			"enter_value": trade.get_enter_value(),
			"current_value": trade.get_current_value()
		}

	def deserialize(self, json_: Dict) -> AgentState.OpenTrade:
		return AgentState.OpenTrade(
			self.__action_serializer.deserialize(json_["trade"]),
			json_["enter_value"],
			json_["current_value"]
		)


class AgentStateSerializer(Serializer):

	def __init__(self):
		super().__init__(AgentState)
		self.__trade_serializer = OpenTradeSerializer()
		self.__market_serializer = MarketStateSerializer()

	def serialize(self, state: AgentState, include_market = False) -> Dict:
		json_ = {
			"balance": state.get_balance(original=True),
			"currency": state._AgentState__currency,
			"margin_rate": state._AgentState__margin_rate,
			"core_pricing": state._AgentState__core_pricing,
			"commission_cost": state._AgentState__commission_cost,
			"open_trades": [
				self.__trade_serializer.serialize(trade)
				for trade in state.get_open_trades()
			]
		}
		if include_market:
			json_["market_state"] = self.__market_serializer.serialize(state._AgentState__market_state)

		return json_

	def deserialize(self, json_: Dict, include_market=False) -> AgentState:

		if include_market:
			json_["market_state"] = self.__market_serializer.deserialize(json_["market_state"])

		return AgentState(
			json_["balance"],
			json_["market_state"],
			margin_rate=json_["margin_rate"],
			currency=json_["currency"],
			core_pricing=json_["core_pricing"],
			commission_cost=json_["commission_cost"],
			open_trades=[
				self.__trade_serializer.deserialize(trade_json)
				for trade_json in json_["open_trades"]
			]
		)


class TradeStateSerializer(Serializer):

	def __init__(self):
		super().__init__(TradeState)
		self.__market_serializer = MarketStateSerializer()
		self.__agent_serializer = AgentStateSerializer()

	def serialize(self, state: TradeState) -> Dict:
		return {
			"market_state": self.__market_serializer.serialize(state.market_state),
			"agent_state": self.__agent_serializer.serialize(state.agent_state, include_market=False),
			"recent_balance": state.get_recent_balance()
		}

	def deserialize(self, json_: Dict) -> TradeState:
		market_state = self.__market_serializer.deserialize(json_["market_state"])
		json_["agent_state"]["market_state"] = market_state
		return TradeState(
			market_state=market_state,
			agent_state=self.__agent_serializer.deserialize(json_["agent_state"], include_market=False),
			recent_balance=json_["recent_balance"]
		)
