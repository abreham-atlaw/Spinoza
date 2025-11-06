import typing
from dataclasses import dataclass

from .trader_action import TraderAction
from .action import Action


@dataclass
class ActionSequence(Action):

	actions: typing.Tuple[TraderAction]

	def __hash__(self):
		return hash(self.actions)
