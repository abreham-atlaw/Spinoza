from abc import ABC
from typing import List

from lib.utils.logger import Logger
from .action_choice_trader import ActionChoiceAgent
from core import Config
from core.agent.action import EndEpisode
from core.environment.trade_state import TradeState


class EpisodicAgent(ActionChoiceAgent, ABC):

	def __init__(self, *args, target_return: float = Config.AGENT_TARGET_RETURN, **kwargs):
		super().__init__(*args, **kwargs)
		self.__target_profit = target_return
		Logger.info(f"Initializing EpisodicAgent with target_profit = {self.__target_profit}")

	def __is_target_achieved(self, state: TradeState) -> bool:
		balance = state.get_agent_state().get_balance()
		initial_balance = state.get_agent_state().initial_balance
		balance_return = balance / initial_balance

		Logger.info(f"[EpisodicAgent] Return: {balance_return}({balance}/{initial_balance})")

		return balance_return >= self.__target_profit

	def _get_optimal_action(self, state, **kwargs):
		if self.__target_profit is not None and self.__is_target_achieved(state):
			return EndEpisode()
		return super()._get_optimal_action(state, **kwargs)
