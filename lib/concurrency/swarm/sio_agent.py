import typing
from abc import ABC, abstractmethod

from socketio import Client


class SIOAgent(ABC):

	def __init__(
			self,
			*args,
			socket_client: Client = None,
			**kwargs
	):
		super().__init__(*args, **kwargs)
		if socket_client is None:
			socket_client = Client(logger=True)
		self._sio = socket_client
		self.__map_events()

	@abstractmethod
	def _map_events(self) -> typing.Dict[str, typing.Callable[[typing.Any], None]]:
		pass

	def __map_events(self):
		for event, handler in self._map_events().items():
			self._sio.on(event, handler)
