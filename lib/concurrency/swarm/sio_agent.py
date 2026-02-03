import typing
from abc import ABC, abstractmethod

from socketio import Client

from lib.concurrency.swarm.swarm_socket import SwarmSocket
from lib.utils.logger import Logger


class SIOAgent(ABC):

	def __init__(
			self,
			*args,
			socket_client: SwarmSocket = None,
			**kwargs
	):
		super().__init__(*args, **kwargs)
		if socket_client is None:
			socket_client = SwarmSocket(logger=True)
		self._sio = socket_client
		self._map_events_map()

	@abstractmethod
	def _map_events(self) -> typing.Dict[str, typing.Callable[[typing.Any], None]]:
		pass

	def __default_map_events(self) -> typing.Dict[str, typing.Callable[[typing.Any], None]]:
		return {
			"disconnect": self._handle_disconnect
		}

	def _map_events_map(self):

		handler_map = self.__default_map_events()
		handler_map.update(self._map_events())

		for event, handler in handler_map.items():
			self._sio.on(event, handler)

	def _handle_disconnect(self):
		Logger.error(f"Socket disconnected.")