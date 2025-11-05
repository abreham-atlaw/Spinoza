from typing import *

import random
import math

from core import Config
from lib.rl.agent import Agent
from lib.utils.logger import Logger
from core.environment.trade_state import TradeState, ArbitradeTradeState
from core.agent.action import TraderAction


class ArbitrageTraderAgent(Agent):

	__STATE_KEY = "ArbitrageTradeState"
	__DIRECTION_ACTION_MAP = {
		-1: TraderAction.Action.SELL,
		1: TraderAction.Action.BUY
	}

	def __init__(
			self,
			zone_size: float = Config.AGENT_ARBITRAGE_ZONE_SIZE,
			zone_guarantee_percent: float = Config.AGENT_ARBITRAGE_ZONE_GUARANTEE_PERCENT,
			base_margin: float = Config.AGENT_ARBITRAGE_BASE_MARGIN,
	):
		super().__init__()
		self.__zone_size = zone_size
		self.__base_margin = base_margin
		self.__zone_guarantee_percent = zone_guarantee_percent

	def _generate_actions(self, state) -> List[object]:
		pass

	def _get_state_action_value(self, state, action, **kwargs) -> float:
		pass

	def _update_state_action_value(self, initial_state, action, final_state, value):
		pass

	@staticmethod
	def __get_crossing_direction(points: Tuple[float, float], point: float) -> Optional[int]:
		for i, cp in enumerate(points):
			d = 2*(i-0.5)
			if d*point - d*cp > 0 or math.isclose(point, cp):
				return int(d)
		return None

	def __generate_arbitrage_state(self, state: TradeState, instrument: Tuple[str, str]) -> ArbitradeTradeState:
		start_point = state.get_market_state().get_current_price(instrument[0], instrument[1])
		real_zone_size = start_point*self.__zone_size
		spread_cost = state.get_market_state().get_spread_state_of(*instrument)
		spread_zone_size = real_zone_size + 2*spread_cost

		checkpoints = start_point-(real_zone_size/2), start_point+(real_zone_size/2)
		spread_checkpoints = checkpoints[0] - spread_cost, checkpoints[1] + spread_cost
		close_points = \
			spread_checkpoints[0] - (spread_zone_size * (1+self.__zone_guarantee_percent)), \
			spread_checkpoints[1] + (spread_zone_size * (1+self.__zone_guarantee_percent))

		return ArbitradeTradeState(
			start_point=start_point,
			checkpoints=checkpoints,
			close_points=close_points,
			instrument=instrument
		)

	def __choose_instrument(self, state: TradeState) -> Tuple[str, str]:
		return random.choice(state.get_market_state().get_tradable_pairs())

	def __is_arbitrage_mode(self, state: TradeState) -> bool:
		return state.is_state_attached(self.__STATE_KEY)

	def __get_arbitrage_state(self, state: TradeState) -> ArbitradeTradeState:
		return state.get_attached_state(self.__STATE_KEY)

	def __detach_arbitrage_state(self, state: TradeState):
		state.detach_state(self.__STATE_KEY)

	def __set_arbitrage_state(self, state: TradeState, arbitrage_state: ArbitradeTradeState):
		state.attach_state(self.__STATE_KEY, arbitrage_state)

	def __start_arbitrage_state(self, state: TradeState) -> Optional[TraderAction]:
		instrument = self.__choose_instrument(state)
		arbitrage_state = self.__generate_arbitrage_state(state, self.__choose_instrument(state))
		Logger.info(f"[+]Starting Arbitrage with {arbitrage_state}")
		self.__set_arbitrage_state(state, arbitrage_state)
		if len(state.get_agent_state().get_open_trades(*instrument)) != 0:
			return TraderAction(
				*instrument,
				TraderAction.Action.CLOSE
			)
		return None

	def __on_insufficient_margin(self, state: TradeState) -> Optional[TraderAction]:
		arbitrage_state = self.__get_arbitrage_state(state)
		self.__detach_arbitrage_state(state)
		return TraderAction(
			*arbitrage_state.instrument,
			TraderAction.Action.CLOSE
		)

	def __on_checkpoint(self, state: TradeState, direction: int) -> Optional[TraderAction]:
		arbitrage_state = self.__get_arbitrage_state(state)
		action = self.__DIRECTION_ACTION_MAP[direction]
		trades = state.get_agent_state().get_open_trades()

		margin_size = None

		if arbitrage_state.margin_stage == ArbitradeTradeState.MarginStage.STAGE_ZERO:
			margin_size = self.__base_margin
		elif action == trades[-1].get_trade().action:
			return None
		elif arbitrage_state.margin_stage == ArbitradeTradeState.MarginStage.STAGE_ONE:
			margin_size = 3 * state.get_agent_state().get_open_trades()[-1].get_trade().margin_used

		arbitrage_state.increment_margin_stage()

		if margin_size > state.get_agent_state().get_margin_available():
			Logger.warning("Insufficient Balance to continue sequence...")
			return self.__on_insufficient_margin(state)

		return TraderAction(
			arbitrage_state.instrument[0],
			arbitrage_state.instrument[1],
			action,
			margin_size
		)

	def __on_close_point(self, state: TradeState, direction: int) -> Optional[TraderAction]:
		arbitrage_state = self.__get_arbitrage_state(state)
		action = TraderAction(
			arbitrage_state.instrument[0],
			arbitrage_state.instrument[1],
			TraderAction.Action.CLOSE,
		)
		self.__detach_arbitrage_state(state)
		return action

	def __monitor_arbitrage(self, state: TradeState) -> Optional[TraderAction]:

		arbitrage_state = self.__get_arbitrage_state(state)

		for points, callback in zip(
				[arbitrage_state.checkpoints, arbitrage_state.close_points],
				[self.__on_checkpoint, self.__on_close_point]
		):
			direction = self.__get_crossing_direction(
				points,
				state.get_market_state().get_current_price(*arbitrage_state.instrument)
			)
			if direction is not None:
				action = callback(state, direction)
				if action is not None:
					return action
		return None

	def _policy(self, state, **kwargs):
		if not self.__is_arbitrage_mode(state):
			return self.__start_arbitrage_state(state)
		return self.__monitor_arbitrage(state)

