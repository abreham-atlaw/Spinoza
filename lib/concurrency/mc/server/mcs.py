import threading
from typing import *
from abc import ABC, abstractmethod

from flask import Flask
from flask_socketio import SocketIO, emit

import time

from lib.rl.agent import MonteCarloAgent, Node
from lib.network.rest_interface.serializers import Serializer
from lib.utils.staterepository import StateRepository
from lib.concurrency.mc.data.staterepository import DistributedStateRepository, FlaskSocketIOChannel
from lib.concurrency.lock import LockManager
from lib.utils.logger import Logger

from temp import stats


class MonteCarloServerAgent(MonteCarloAgent, ABC):

	def select(self, parent_state_node: 'Node', lock_checker) -> Union[None, 'Node']:
		prioritized_children = sorted(
			[node for node in parent_state_node.get_children() if not lock_checker(node)],
			key=self._uct,
			reverse=True
		)

		for child in prioritized_children:
			chosen_state_node: Node = self._get_random_state_node(child)
			if not chosen_state_node.has_children():
				return chosen_state_node

			leaf_node = self.select(chosen_state_node, lock_checker)
			if leaf_node is not None:
				return leaf_node

		return None

	def backpropagate(self, node: Node):
		self._backpropagate(node)



class PassThroughSerializer(Serializer):

	def __init__(self):
		super(PassThroughSerializer, self).__init__(None)

	def serialize(self, data: object):
		return data

	def deserialize(self, json_: Dict):
		return json_


class MonteCarloServer(ABC):

	def __init__(self, sleep_time=0.01, host="127.0.0.1", port=8000):
		self.__app = Flask(__name__)
		self._socketio = SocketIO(self.__app)
		self.__agent = self._init_agent()
		self.__current_graph = None
		self.__graph_serializer = self._init_graph_serializer()
		self.__state_serializer = self._init_state_serializer()
		self.__repository = self._init_state_repository()
		self.__agent.set_repository(self.__repository)
		self.__locked_nodes: List[str] = []
		self.__active = False
		self.__sleep_time = sleep_time
		self.__lock_manager = LockManager()
		self.__host = host
		self.__port = port

	@abstractmethod
	def _init_agent(self) -> MonteCarloServerAgent:
		pass

	@abstractmethod
	def _init_graph_serializer(self) -> Serializer:
		pass

	def _init_state_repository(self) -> StateRepository:
		return DistributedStateRepository(
			FlaskSocketIOChannel(
				self._socketio
			),
			self.__state_serializer,
			is_server=True
		)

	def _init_state_serializer(self) -> Serializer:
		return PassThroughSerializer()

	def _map_events(self) -> List[Tuple[str, object]]:
		return [
			("new", self.__handle_new),
			("select", self.__handle_select),
			("backpropagate", self.__handle_backpropagate),
			("end", self.__handle_end),
		]

	def _set_graph(self, graph: Node):
		self.__current_graph = graph

	def _get_graph(self) -> Node:
		return self.__current_graph

	def _set_active(self, active):
		self.__active = active
		if active:
			emit("new", broadcast=True)

		if not active:
			self.__repository.clear()
			emit("end", broadcast=True)

	def _is_active(self):
		return self.__active

	def __map_events(self):
		for event, handler in self._map_events():
			self._socketio.on_event(event, handler)

	def __is_locked(self, node: Node):
		return node.id in self.__locked_nodes
		# return self.__lock_manager.lock_and_do(
		# 	var=self.__locked_nodes,
		# 	func=lambda: node.id in self.__locked_nodes
		# )

	def __lock(self, node: Node):
		self.__locked_nodes.append(node.id)
		# self.__lock_manager.lock_and_do(
		# 	var=self.__locked_nodes,
		# 	func=lambda: self.__locked_nodes.append(node.id)
		# )

	def __unlock(self, node: Node):
		self.__locked_nodes.remove(node.id)
		# self.__lock_manager.lock_and_do(
		# 	var=self.__locked_nodes,
		# 	func=lambda : self.__locked_nodes.remove(node.id)
		# )

	def __select(self):
		leaf_node = None
		while leaf_node is None:
			if not self._is_active():
				return

			leaf_node = self.__agent.select(self._get_graph(), self.__is_locked)
			if not self._get_graph().has_children() and not self.__is_locked(self._get_graph()):
				leaf_node = self._get_graph()

			if leaf_node is None:
				time.sleep(self.__sleep_time)

		self.__lock(leaf_node)

		return leaf_node

	def __handle_new(self, state):
		print("Received New Request")
		self._set_graph(Node(None, None, Node.NodeType.STATE))
		self.__repository.store(self._get_graph().id, self.__state_serializer.deserialize(state))
		self._set_active(True)

	def __handle_select(self):
		if not self._is_active():
			emit("end")
			return

		leaf_node = self.__lock_manager.lock_and_do(
			var=self.__locked_nodes,
			func=self.__select
		)

		if leaf_node is None:
			return

		emit("select", self.__graph_serializer.serialize_json(leaf_node), broadcast=False)

	def __handle_backpropagate(self, node):
		node: Node = self.__graph_serializer.deserialize_json(node)
		parent = self._get_graph().find_node_by_id(node.id).parent
		if parent is None:
			self._set_graph(node)
		else:
			parent.children.remove(node)
			parent.add_child(node)
			self.__agent.backpropagate(node)
		self.__lock_manager.lock_and_do(
			var=self.__locked_nodes,
			func=lambda: self.__unlock(node)
		)

	def __handle_end(self):
		self._set_active(False)
		Logger.info(
			f"Simulations Done: "
			f"Depth: {stats.get_max_depth(self._get_graph())}, "
			f"Nodes: {len(stats.get_nodes(self._get_graph()))}"
		)
		optimal_action_node = max(self._get_graph().get_children(), key=lambda node: node.get_total_value())
		emit("action", self.__graph_serializer.serialize(optimal_action_node)["action"])

	def start(self):
		self.__map_events()
		self._socketio.run(self.__app, host=self.__host, port=self.__port)
