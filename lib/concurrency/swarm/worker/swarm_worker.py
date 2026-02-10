import random
import typing
from abc import ABC

from socketio.exceptions import BadNamespaceError

from lib.concurrency.swarm.sio_agent import SIOAgent
from lib.network.rest_interface import Serializer
from lib.rl.agent import MonteCarloAgent, Node
from lib.utils.decorators import handle_exception
from lib.utils.logger import Logger


class SwarmWorker(SIOAgent, MonteCarloAgent, ABC):

	def __init__(
			self,
			*args,
			node_serializer: Serializer[Node],
			**kwargs
	) -> None:
		super().__init__(*args, **kwargs)
		self.__node_serializer = node_serializer

	def _map_events(self) -> typing.Dict[str, typing.Callable[[typing.Any], None]]:
		return {
			"select": self.__handle_select
		}

	def __handle_select(self, data = None):
		if data is None:
			Logger.error(f"Received select with empty data.")
			self._emit_select()
			return

		node = self.__node_serializer.deserialize(data)
		Logger.info(f"Working on node: {node.id}")

		self._monte_carlo_simulation(node)

		self.__backpropagate(node)
		self._emit_select()

	@handle_exception(exception_cls=(BadNamespaceError,))
	def _emit_select(self):
		self._sio.emit("select")

	@handle_exception(exception_cls=(BadNamespaceError,))
	def __backpropagate(self, node: Node):
		Logger.info(f"Backpropagating node: {node.id}")
		self._sio.emit(
			"backpropagate",
			self.__node_serializer.serialize(node),
		)

	def perform_timestep(self):
		self._emit_select()
		self._sio.wait()
