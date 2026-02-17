import typing
from abc import ABC

from lib.utils.logger import Logger
from .action_choice_trader import ActionChoiceAgent
from core import Config
from core.agent.action import EndEpisode, Action, ActionSequence, TraderAction
from core.environment.trade_state import TradeState


class EpisodicAgent(ActionChoiceAgent, ABC):

	def __init__(
			self,
			*args,
			episode_take_profit: float = Config.AGENT_EPISODE_TAKE_PROFIT,
			episode_stop_loss: float = Config.AGENT_EPISODE_STOP_LOSS,
			attach_take_profit: bool = Config.AGENT_EPISODE_ATTACH_TAKE_PROFIT,
			**kwargs):
		super().__init__(*args, **kwargs)
		self.__take_profit = episode_take_profit
		self.__stop_loss = episode_stop_loss
		self.__use_attach_take_profit = attach_take_profit
		Logger.info(
			f"Initializing EpisodicAgent with target_profit = {self.__take_profit}, stop_loss = {self.__stop_loss}, "
			f"attach_take_profit = {self.__use_attach_take_profit}"
		)

	def __is_target_achieved(self, state: TradeState) -> bool:
		balance = state.get_agent_state().get_balance()
		initial_balance = state.get_agent_state().initial_balance
		balance_return = balance / initial_balance

		Logger.info(f"[EpisodicAgent] Return: {balance_return}({balance}/{initial_balance})")

		return (self.__take_profit is not None and balance_return >= self.__take_profit) or (self.__stop_loss is not None and balance_return <= self.__stop_loss)

	def __attach_take_profit(self, action: typing.Union[Action, None], state: TradeState) -> Action:
		if action is None:
			return action

		if isinstance(action, ActionSequence):
			self.__attach_take_profit(action.actions[-1], state)
			return action

		assert isinstance(action, TraderAction)

		if action.action == TraderAction.Action.CLOSE or action.take_profit is not None:
			return action

		take_profit = (
			(
					(
							self.__take_profit /
							(state.get_agent_state().get_balance() / state.get_agent_state().initial_balance)
					) - 1
			) * (
				state.get_agent_state().get_balance() * state.get_agent_state().get_margin_rate() / action.margin_used
			)
		) + 1
		if action.action == TraderAction.Action.SELL:
			take_profit = 1/take_profit

		Logger.info(f"[EpisodicAgent] Attaching Take Profit: {take_profit} on action: {action}...")
		action.take_profit = take_profit

		return action

	def _get_optimal_action(self, state, **kwargs):
		if self.__take_profit is not None and self.__is_target_achieved(state):
			return EndEpisode()
		action = super()._get_optimal_action(state, **kwargs)
		if self.__use_attach_take_profit:
			action = self.__attach_take_profit(action, state)

		return action
