from abc import ABC

from lib.utils.logger import Logger
from lib.utils.stm import ShortTermMemory
from .mca import MonteCarloAgent
from .node import Node


class ReflexMonteCarloAgent(MonteCarloAgent, ABC):

	def __init__(
			self,
			reflex_stm: ShortTermMemory,
			*args,
			**kwargs
	):
		super().__init__(
			*args, **kwargs
		)
		self.__reflex_stm = reflex_stm

	def __get_approximate_node(self, state) -> Node:
		self.__reflex_stm.clear()

		state_nodes = self._get_current_graph().get_children()[0].get_children()

		for state_node in state_nodes:
			state = self._state_repository.retrieve(state_node.id)
			self.__reflex_stm.memorize(state)

		approx_state = self.__reflex_stm.recall(state)

		if approx_state is None:
			raise ValueError(f"Reflex STM returned None.")

		node = next(filter(
			lambda sn: self._state_repository.retrieve(sn.id) is approx_state,
			state_nodes
		))

		return node

	def _prepare_reflex_action(self):
		Logger.info(f"Setting Reflex Node")
		new_state = self._get_environment().get_latest_state()

		if len(self._get_current_graph().get_children()) > 1:
			raise ValueError("Found more than 1 action node on root node while using reflex mode.")

		node = self.__get_approximate_node(new_state)
		self._set_current_graph(node)


	def _monte_carlo_tree_search(self, state) -> None:
		super()._monte_carlo_tree_search(state)
		self._prepare_reflex_action()

