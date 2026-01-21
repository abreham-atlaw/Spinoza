import json
import os
import random
import time
import typing
from datetime import datetime
from typing import *
from abc import ABC

import numpy as np

import gc
import psutil

from lib.network.rest_interface import Serializer
from lib.rl.agent import ModelBasedAgent
from lib.rl.environment import ModelBasedState
from lib.utils.logger import Logger
from lib.utils.math import sigmoid
from lib.utils.staterepository import StateRepository, SectionalDictStateRepository
from lib.utils.stm import ShortTermMemory
from temp import stats
from .node import Node
from .stm.node_memory_matcher import NodeMemoryMatcher
from .stm.node_stm import NodeShortTermMemory
from .resource_manager import MCResourceManager


class MonteCarloAgent(ModelBasedAgent, ABC):

	def __init__(
			self,
			*args,
			resource_manager: MCResourceManager,
			min_free_memory_percent=10,
			logical=False,
			uct_exploration_weight=1,
			state_repository: StateRepository = None,
			use_stm: bool = True,
			short_term_memory: NodeShortTermMemory = None,
			probability_correction: bool = False,
			min_probability: typing.Optional[float] = None,
			top_k_nodes: typing.Optional[int] = None,
			dynamic_k_threshold: float = None,
			dump_nodes: bool = False,
			dump_path: str = "./",
			dump_visited_only: bool = False,
			node_serializer: typing.Optional['NodeSerializer'] = None,
			state_serializer: typing.Optional[Serializer] = None,
			squash_epsilon: float = 1e-9,
			graph_plot_rate: float = 0.0,
			**kwargs
	):

		super(MonteCarloAgent, self).__init__(*args, **kwargs)
		self.__min_free_memory = min_free_memory_percent
		self.__logical = logical
		self.__uct_exploration_weight = uct_exploration_weight
		self.__set_mode(logical)
		self.__min_probability = min_probability
		self.__use_stm: bool = use_stm
		self.__short_term_memory = short_term_memory
		if short_term_memory is None and use_stm:
			self.__short_term_memory = NodeShortTermMemory(
				size=10,  # TODO: GET SIZE FROM FIRST INSTANCE OF RUNNING
				matcher=NodeMemoryMatcher()
			)
		self._state_repository = state_repository
		if state_repository is None:
			self._state_repository = SectionalDictStateRepository(2, 15, serializer=state_serializer)
		self.__short_term_memory.get_matcher().set_repository(self._state_repository)
		self.__probability_correction = probability_correction
		self.__current_graph_: Optional[Node] = None
		self.__top_k_nodes = top_k_nodes
		self.__dump_nodes = dump_nodes
		self.__dump_path = dump_path
		self.__dump_visited_only = dump_visited_only
		self.__serializer = node_serializer
		self.__squash_epsilon = squash_epsilon
		self.__dynamic_k_threshold = dynamic_k_threshold
		self.__resource_manager = resource_manager
		self.__graph_plot_rate = graph_plot_rate

	@property
	def trim_mode(self) -> bool:
		return self.__top_k_nodes is not None or self.__dynamic_k_threshold is not None

	def set_repository(self, repository: StateRepository):
		self._state_repository = repository

	def __set_mode(self, logical: bool):
		if logical:
			self._backpropagate, self._uct = self.__logical_backpropagate, self.__logical_uct
		else:
			self._backpropagate, self._uct = self.__legacy_backpropagate, self.__legacy_uct

	def _init_resources(self) -> object:
		return self.__resource_manager.init_resource()

	def _has_resource(self, resources) -> bool:
		return self.__resource_manager.has_resource(resources)

	def _set_current_graph(self, graph: 'Node'):
		self.__current_graph = graph

	def _get_current_graph(self) -> 'Node':
		if self.__current_graph is None:
			raise ValueError("Graph not set. Make sure __init_current_graph was called.")
		return self.__current_graph

	def _get_state_value(self, state):
		return self._get_environment().get_reward(state)

	def _get_random_state_node(self, action_node: 'Node') -> 'Node':
		state_nodes: List[Node] = action_node.get_children()

		if len(state_nodes) == 1:
			return state_nodes[0]

		probabilities = np.array([
			state_node.weight
			for state_node in state_nodes
		]).astype('float64')
		counts = np.array([
			state_node.visits
			for state_node in state_nodes
		]).astype('float64')

		if self.__probability_correction:
			probabilities = self.__correct_probabilities(probabilities, counts)
		else:
			probabilities = self.__squash_probabilities(probabilities)

		if self.__min_probability is not None:
			probabilities[probabilities < self.__min_probability] = 0.0
			probabilities = self.__squash_probabilities(probabilities)

		# noinspection PyTypeChecker
		choice: Node = np.random.choice(
			state_nodes,
			1,
			p=probabilities
		)[0]

		return choice

	def __squash_probabilities(self, probs: np.ndarray) -> np.ndarray:
		probs[probs <= 0] = self.__squash_epsilon
		return probs / probs.sum()

	def __correct_probabilities(self, expected: np.ndarray, counts: np.ndarray):
		total_counts = counts.sum()

		if total_counts == 0:
			return expected

		frequency_probs = counts / counts.sum()

		corrected = self.__squash_probabilities(
			(2 * expected) - frequency_probs
		)
		return corrected

	def __manage_resources(self, end=False):
		if psutil.virtual_memory().percent > (100 - self.__min_free_memory) or end:
			Logger.info("Realising Memory. Calling gc.collect")
			gc.collect()

	@staticmethod
	def __legacy_uct(node: 'Node') -> float:
		if not node.is_visited():
			return np.inf

		return (node.get_total_value()/node.get_visits()) + np.sqrt(np.log(node.parent.get_visits())/node.get_visits())

	def __logical_uct(self, node: 'Node') -> float:
		if not node.is_visited():
			return np.inf

		return (
				sigmoid(node.get_total_value()) +
				np.sqrt(np.log(node.parent.get_visits())/node.get_visits()) *
				self.__uct_exploration_weight
		)

	def __check_stm(self, node: 'Node') -> Optional['Node']:
		if self.__short_term_memory is None:
			return None

		memory: Optional[Node] = self.__short_term_memory.recall(node)
		if memory is None:
			return None
		return memory

	@staticmethod
	def __move_node(source: 'Node', destination: 'Node'):
		destination.children = source.children
		destination.visits = source.visits
		destination.total_value = source.total_value
		destination.weight = source.weight
		for child in source.children:
			child.parent = destination

	def __get_descendant_state_nodes(self, node: Node) -> typing.List[Node]:
		descendants = []
		for action_node in node.get_children():
			descendants.extend(action_node.get_children())
			for state_node in action_node.get_children():
				descendants.extend(self.__get_descendant_state_nodes(state_node))
		return descendants

	def __sync_repository(self):
		Logger.info(f"Syncing state repository({len(self._state_repository)} states present)...")
		descendants = [self._get_current_graph()] + self.__get_descendant_state_nodes(self._get_current_graph())
		state_backups = {node.id: self._state_repository.retrieve(node.id) for node in descendants}
		self._state_repository.clear()
		for id_, state in state_backups.items():
			self._state_repository.store(id_, state)
		Logger.info(f"Synced state repository({len(self._state_repository)} states present).")

	def __clear_stm(self):
		Logger.info("Clearing short-term memory...")
		self.__short_term_memory.clear()

	def __expand_from_stm(self, node: 'Node') -> bool:

		memorized = self.__check_stm(node)

		resolved = False

		if memorized is not None and memorized.has_children():
			self.__move_node(memorized, node)
			resolved = True
		self.__sync_repository()
		if self.__use_stm:
			self.__clear_stm()

		return resolved

	def __store_to_stm(self, root: 'Node'):

		state_nodes = []
		for action_node in root.get_children():
			state_nodes.extend(action_node.get_children())

		for node in sorted(state_nodes, key=lambda n: n.weight, reverse=True)[:self.__short_term_memory.size]:
			self.__short_term_memory.memorize(node)

	def __get_children_states(self, state_node: 'Node') -> Dict[str, object]:
		if not state_node.has_children():
			return {state_node.id: self._state_repository.retrieve(state_node.id)}
		states = {}
		for action_node in state_node.get_children():
			for child in action_node.get_children():
				states.update(self.__get_children_states(child))

		return states

	def __backup_states(self) -> Dict[str, object]:
		states = {}
		for memory in self.__short_term_memory:
			states[memory.id] = self._state_repository.retrieve(memory.id)
			states.update(self.__get_children_states(memory))

		return states

	def __restore_backups(self, backup: Dict[str, object]):
		for key, state in backup.items():
			self._state_repository.store(key, state)

	def __backup_and_clear_repository(self):
		backup = self.__backup_states()
		self._state_repository.clear()
		self.__restore_backups(backup)

	def __dump_node(self, node):
		dump_path = os.path.abspath(os.path.join(self.__dump_path, f"{datetime.now().timestamp()}"))
		os.mkdir(dump_path)

		json_ = self.__serializer.serialize(node)
		with open(os.path.join(dump_path, "graph.json"), "w") as file:
			json.dump(json_, file)

		self._state_repository.dump(
			os.path.join(dump_path, "states.json"),
			keys=[
				node.id
				for node in stats.get_nodes(node, visited=True, include_root=True)
				if node.node_type == Node.NodeType.STATE
			]
		)

	def _plot_node(self, node: Node):
		if self.__graph_plot_rate > 0 and random.random() < self.__graph_plot_rate:
			stats.draw_graph_live(node, visited=True, state_repository=self._state_repository, uct_fn=self._uct)

	def _finalize_step(self, root: 'Node'):

		if self.__dump_nodes:
			self.__dump_node(root)

		if self.__use_stm:
			self.__store_to_stm(root)
			# self.__backup_and_clear_repository()
		else:
			self._state_repository.clear()
		self.__manage_resources(end=True)

	def _select(self, parent_state_node: 'Node') -> 'Node':

		promising_action_node: Node = max(
													parent_state_node.get_children(),
													key=lambda node: self._uct(node)
												)

		chosen_state_node: Node = self._get_random_state_node(promising_action_node)

		if len(chosen_state_node.get_children()) == 0:
			return chosen_state_node

		return self._select(chosen_state_node)

	def __collect_transition_inputs(
			self,
			state_node: 'Node'
	) -> Tuple[List[ModelBasedState], List[Any], List[ModelBasedState], List['Node']]:
		states_inputs = []
		actions_inputs = []
		final_states_inputs = []
		state_nodes = []
		# noinspection PyTypeChecker
		state: ModelBasedState = self._state_repository.retrieve(state_node.id)
		for action in self._get_available_actions(state):
			action_node = Node(state_node, action, Node.NodeType.ACTION)
			for final_state in self._get_possible_states(state, action):
				states_inputs.append(state)
				actions_inputs.append(action)
				final_states_inputs.append(final_state)

				final_state_node = Node(
					action_node,
					None,
					Node.NodeType.STATE,
				)
				action_node.add_child(final_state_node)
				state_nodes.append(final_state_node)
			state_node.add_child(action_node)

		return states_inputs, actions_inputs, final_states_inputs, state_nodes

	def __get_k(self, node: 'Node') -> int:
		if not self.__dynamic_k_threshold:
			return self.__top_k_nodes
		weights = np.array([child.weight for child in node.get_children()])
		weights = weights / np.sum(weights)
		importance = weights / np.max(weights)
		return np.sum(importance > self.__dynamic_k_threshold)

	def __trim_node(self, node: 'Node'):
		k = self.__get_k(node)
		node.children = sorted(node.children, key=lambda n: n.weight, reverse=True)[:k]

	def _expand(self, state_node: 'Node', stm=True):
		if self._get_environment().is_episode_over(self._state_repository.retrieve(state_node.id)):
			return

		if self.__use_stm and stm and self.__expand_from_stm(state_node):
			Logger.info("Recalled from STM")
			Logger.info(f"Recall Depth: {stats.get_max_depth(state_node)}")
			return

		states, actions, final_states, possible_state_nodes = self.__collect_transition_inputs(state_node)

		start_time = datetime.now()
		probability_distribution = self._get_expected_transition_probability_distribution(states, actions, final_states)
		stats.durations["get_expected_transition_probability_distribution"] += (datetime.now() - start_time).total_seconds()

		start_time = datetime.now()
		for i in range(len(probability_distribution)):
			possible_state_nodes[i].total_value = possible_state_nodes[i].instant_value = self._get_expected_instant_reward(final_states[i])
			possible_state_nodes[i].weight = probability_distribution[i]
			possible_state_nodes[i].depth = state_node.depth + 1
			self._state_repository.store(possible_state_nodes[i].id, final_states[i])
			final_states[i].set_depth(possible_state_nodes[i].depth)
		stats.durations["for i in range(len(probability_distribution))"] += (
						datetime.now() - start_time).total_seconds()

		start_time = datetime.now()
		if self.trim_mode:
			for action_node in state_node.children:
				self.__trim_node(action_node)
		stats.durations["trim"] += (datetime.now() - start_time).total_seconds()

	def __simulate(self, state_node: 'Node') -> 'Node':

		if not state_node.has_children():
			return state_node

		action_node: Node = np.random.choice(state_node.get_children(), 1)[0]

		return self._get_random_state_node(action_node)

	def _get_action_node_value(self, node: 'Node'):
		total_weight = np.sum([state_node.weight for state_node in node.get_children()])
		return np.sum([
			state_node.get_total_value() * state_node.weight / total_weight
			for state_node in node.get_children()
		])

	def __legacy_backpropagate(self, node: 'Node', reward=None) -> None:
		node.increment_visits()
		if node.parent is None:
			return

		if reward is None:
			reward = node.instant_value
		node.add_value(self._get_discount_factor(node.depth) * reward)

		if node.node_type == Node.NodeType.ACTION:
			reward = reward * self._get_discount_factor(node.depth)

		self.__legacy_backpropagate(node.parent, reward)

	def __logical_backpropagate(self, node: 'Node') -> None:
		node.increment_visits()

		if node.node_type == Node.NodeType.STATE:
			node.set_total_value(node.instant_value)
			if node.has_children():
				node.add_value(float(
					self._get_discount_factor(node.depth) *
					max([action_node.get_total_value() for action_node in node.get_children()])
				))

		else:
			node.set_total_value(
				self._get_action_node_value(node)
			)

		if node.parent is None:
			return

		self.__logical_backpropagate(node.parent)

	def _monte_carlo_simulation(self, root_node: 'Node'):
		self._expand(root_node)

		resources = self._init_resources()

		stats.iterations["main_loop"] = 0
		while self._has_resource(resources):
			start_time = datetime.now()
			leaf_node = self._select(root_node)
			stats.durations["select"] += (datetime.now() - start_time).total_seconds()

			start_time = datetime.now()
			self._expand(leaf_node, stm=False)
			stats.durations["expand"] += (datetime.now() - start_time).total_seconds()

			start_time = datetime.now()
			final_node = self.__simulate(leaf_node)
			stats.durations["simulate"] += (datetime.now() - start_time).total_seconds()

			self._backpropagate(final_node)

			self.__manage_resources()
			self._plot_node(root_node)

			stats.iterations["main_loop"] += 1

	def _monte_carlo_tree_search(self, state) -> None:
		root_node = Node(None, None, Node.NodeType.STATE)
		self._state_repository.store(root_node.id, state)
		self._set_current_graph(root_node)

		self._monte_carlo_simulation(root_node)
		root_node = self._get_current_graph()

		Logger.info(
			f"Simulations Done: Iterations: {stats.iterations['main_loop']}, "
			f"Depth: {stats.get_max_depth(root_node)}, "
			f"Nodes: {len(stats.get_nodes(root_node))}"
		)
		optimal_action = max(root_node.get_children(), key=lambda node: node.get_total_value()).action
		Logger.info(f"Best Action {optimal_action}")
		self._finalize_step(root_node)

	def _get_state_action_value(self, state, action, **kwargs) -> float:
		for action_node in self._get_current_graph().get_children():
			if action_node.action == action:
				return action_node.get_total_value()

		raise Exception(f"Action Not Found in Graph.\nGiven Action={action}.\nAvailable Actions={[node.action for node in self._get_current_graph().get_children()]}")

	def _get_optimal_action(self, state, **kwargs):
		self._monte_carlo_tree_search(state)
		return super()._get_optimal_action(state, **kwargs)

	def _explore(self, state):
		resource = self._init_resources()
		while self._has_resource(resource):
			time.sleep(0.5)
		return super()._explore(state)
