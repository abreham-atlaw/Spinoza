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

	def _map_events(self) -> typing.Dict[str, typing.Callable[[typing.Any], None]]:
		return {
			"backpropagate": self.__handle_backpropagate,
		}

	def __queue_node(self, node: Node):
		self._sio.emit(
			"queue",
			data=self.__node_serializer.serialize(node)
		)

	def __handle_backpropagate(self, data = None):
		if data is None:
			Logger.error(f"Received data=None on backpropagate")
			return
		node: Node = self.__node_serializer.deserialize(data)
		parent = self._get_current_graph().find_node_by_id(node.id).parent

		parent.children.remove(node)
		parent.add_child(node)
		self._backpropagate(node)

	def _finalize_step(self, root: 'Node'):
		super()._finalize_step(root)
		self.__queued_nodes = []

	def _monte_carlo_loop(self, root_node: Node):

		leaf_node = self._select(root_node)

		if leaf_node not in self.__queued_nodes:
			self.__queue_node(leaf_node)
			self.__queued_nodes.append(leaf_node)

		time.sleep(self.__queue_wait_time)

