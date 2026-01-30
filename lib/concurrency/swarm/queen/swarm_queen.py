import time
import typing
from abc import ABC

from lib.concurrency.swarm.sio_agent import SIOAgent
from lib.network.rest_interface import Serializer
from lib.rl.agent import MonteCarloAgent, Node
from lib.utils.logger import Logger


class SwarmQueen(SIOAgent, MonteCarloAgent, ABC):

	def __init__(
			self,
			*args,
			node_serializer: Serializer,
			queue_wait_time: float = 0.5,
			**kwargs
	):
		super().__init__(*args, **kwargs)
		self.__node_serializer = node_serializer
		self.__queue_wait_time = queue_wait_time

		self.__queued_nodes = []
		self.__is_active = False

	def _map_events(self) -> typing.Dict[str, typing.Callable[[typing.Any], None]]:
		return {
			"backpropagate": self.__handle_backpropagate,
		}

	def __queue_node(self, node: Node):
		self._sio.emit(
			"queue",
			data=self.__node_serializer.serialize(node)
		)

	def __clear_queue(self):
		Logger.info(f"Clearing Queue...")
		self._sio.emit(
			"clear-queue"
		)

	def __handle_backpropagate(self, data = None):
		if not self.__is_active:
			Logger.warning(f"Received Backpropagate while inactive. Skipping...")
			return

		if data is None:
			Logger.error(f"Received data=None on backpropagate")
			return

		node: Node = self.__node_serializer.deserialize(data)
		Logger.info(f"Backpropagating node: {node.id}")
		parent = self._get_current_graph().find_node_by_id(node.id).parent

		parent.children.remove(node)
		parent.add_child(node)
		self._backpropagate(node)

	def _finalize_step(self, root: 'Node'):
		self.__deactivate_simulation()
		super()._finalize_step(root)

	def __activate_simulation(self):
		self.__is_active = True

	def __deactivate_simulation(self):
		self.__is_active = False
		self.__clear_queue()
		self.__queued_nodes = []

	def _monte_carlo_loop(self, root_node: Node):

		leaf_node = self._select(root_node)

		if leaf_node not in self.__queued_nodes:
			self.__queue_node(leaf_node)
			self.__queued_nodes.append(leaf_node)

		time.sleep(self.__queue_wait_time)

	def _monte_carlo_simulation(self, root_node: 'Node'):
		self.__activate_simulation()
		return super()._monte_carlo_simulation(root_node)
