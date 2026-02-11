import time
import typing
from abc import ABC
from datetime import datetime

from socketio.exceptions import BadNamespaceError

from lib.concurrency.swarm.sio_agent import SIOAgent
from lib.network.rest_interface import Serializer
from lib.rl.agent import MonteCarloAgent, Node
from lib.utils.decorators import handle_exception, retry
from lib.utils.logger import Logger


class SwarmQueen(SIOAgent, MonteCarloAgent, ABC):

	def __init__(
			self,
			*args,
			node_serializer: Serializer,
			queue_timeout: float,
			queue_wait_time: float = 0.5,
			**kwargs
	):
		super().__init__(*args, **kwargs)
		self.__node_serializer = node_serializer
		self.__queue_wait_time = queue_wait_time

		self.__queued_nodes = []
		self.__queue_time = {}
		self.__is_active = False
		self.__queue_timeout = queue_timeout

	def _map_events(self) -> typing.Dict[str, typing.Callable[[typing.Any], None]]:
		return {
			"backpropagate": self.__handle_backpropagate,
		}

	def __get_queue_time(self, node: Node) -> datetime:
		return self.__queue_time.get(node.id)

	def __set_queue_time(self, node: Node):
		self.__queue_time[node.id] = datetime.now()

	@handle_exception(exception_cls=(BadNamespaceError,))
	def __queue_node(self, node: Node):
		self._sio.emit(
			"queue",
			data=self.__node_serializer.serialize(node)
		)
		self.__set_queue_time(node)

	@handle_exception(exception_cls=(BadNamespaceError,))
	@retry(exception_cls=(BadNamespaceError,), sleep_timer=10, patience=10)
	def __clear_queue(self):
		Logger.info(f"Clearing Queue...")
		self._sio.emit(
			"clear-queue"
		)

	def __monitor_queue_timeouts(self):
		for node in self.__queued_nodes:
			if (datetime.now() - self.__get_queue_time(node)).total_seconds() > self.__queue_timeout:
				Logger.info(f"Re-Queueing Node: {node.id}")
				self.__queue_node(node)

	def __handle_backpropagate(self, data = None):
		if not self.__is_active:
			Logger.warning(f"Received Backpropagate while inactive. Skipping...")
			return

		if data is None:
			Logger.error(f"Received data=None on backpropagate")
			return

		node: Node = self.__node_serializer.deserialize(data)
		Logger.info(f"Backpropagating node: {node.id}")

		old_node = self._get_current_graph().find_node_by_id(node.id)
		if old_node is None:
			Logger.warning(f"Received Backpropagate to an unknown node. Skipping...")
			return

		parent = old_node.parent

		parent.children.remove(node)
		parent.add_child(node)
		self._backpropagate(node)
		if old_node in self.__queued_nodes:
			self.__queued_nodes.remove(old_node)

	def _finalize_step(self, root: 'Node'):
		self._deactivate_simulation()
		super()._finalize_step(root)

	def _activate_simulation(self):
		self.__is_active = True

	def _deactivate_simulation(self):
		self.__is_active = False
		self.__clear_queue()
		self.__queued_nodes = []

	def _monte_carlo_loop(self, root_node: Node):

		if not self.__is_active:
			time.sleep(self.__queue_wait_time)
			return

		leaf_node = self._select(root_node)

		if leaf_node not in self.__queued_nodes:
			self.__queue_node(leaf_node)
			self.__queued_nodes.append(leaf_node)

		self.__monitor_queue_timeouts()
		time.sleep(self.__queue_wait_time)

	def _monte_carlo_simulation(self, root_node: 'Node'):
		self._activate_simulation()
		return super()._monte_carlo_simulation(root_node)
