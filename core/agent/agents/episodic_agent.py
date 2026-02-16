from abc import ABC

from lib.utils.logger import Logger
from .action_choice_trader import ActionChoiceAgent
from core import Config
from core.agent.action import EndEpisode
from core.environment.trade_state import TradeState


class EpisodicAgent(ActionChoiceAgent, ABC):

	def __init__(
			self,
			*args,
			episode_take_profit: float = Config.AGENT_EPISODE_TAKE_PROFIT,
			episode_stop_loss: float = Config.AGENT_EPISODE_STOP_LOSS,
			**kwargs):
		super().__init__(*args, **kwargs)
		self.__take_profit = episode_take_profit
		self.__stop_loss = episode_stop_loss
		Logger.info(f"Initializing EpisodicAgent with target_profit = {self.__take_profit}, stop_loss = {self.__stop_loss}")

	def __is_target_achieved(self, state: TradeState) -> bool:
		balance = state.get_agent_state().get_balance()
		initial_balance = state.get_agent_state().initial_balance
		balance_return = balance / initial_balance

		Logger.info(f"[EpisodicAgent] Return: {balance_return}({balance}/{initial_balance})")

		return (balance_return >= self.__take_profit) or (balance_return <= self.__stop_loss)

	def _get_optimal_action(self, state, **kwargs):
		if self.__take_profit is not None and self.__is_target_achieved(state):
			return EndEpisode()
		return super()._get_optimal_action(state, **kwargs)
