import typing
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

	def _search_approximate_node(self, state, state_nodes: typing.List[Node]):
		self.__reflex_stm.clear()

		states = [
			self._state_repository.retrieve(node.id)
			for node in state_nodes
		]

		if len(state_nodes) == 0:
			Logger.error(f"State Node is empty. Using original node")
			return self._get_current_graph()

		for i in range(len(states)):
			self.__reflex_stm.memorize(states[i])

		approx_state = self.__reflex_stm.recall(state)

		if approx_state is None:
			raise ValueError(f"Reflex STM returned None.")

		idx = states.index(approx_state)
		node = state_nodes[idx]

		Logger.info(f"Approximating to node {idx}")

		return node

	def _approximate_node(self, state) -> Node:
		state_nodes = self._get_current_graph().get_children()[0].get_children()
		return self._search_approximate_node(state, state_nodes)

	def _prepare_reflex_action(self):
		Logger.info(f"Setting Reflex Node")
		new_state = self._get_environment().get_latest_state()

		if len(self._get_current_graph().get_children()) > 1:
			raise ValueError("Found more than 1 action node on root node while using reflex mode.")

		node = self._approximate_node(new_state)
		self._set_current_graph(node)

	def _monte_carlo_simulation(self, root_node: 'Node'):
		super()._monte_carlo_simulation(root_node)
		self._prepare_reflex_action()

	def _get_optimal_action(self, state, **kwargs):
		self._monte_carlo_tree_search(state)
		if len(self._get_current_graph().get_children()) == 0:
			Logger.warning(f"State Node is empty. Returning None.")
			return None
		return max(self._get_current_graph().get_children(), key=lambda node: node.get_total_value()).action

	def _finalize_step(self, root: 'Node'):
		super()._finalize_step(root.parent.parent)
