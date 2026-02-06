import typing

import socketio

from lib.utils.logger import Logger


class SwarmSocket:

	def __init__(self, *args, **kwargs):
		self.__args = args
		self.__kwargs = kwargs
		self.__handlers = {}
		self.__sio = self.__init_sio()

	def __init_sio(self) -> socketio.Client:
		client = socketio.Client(*self.__args, **self.__kwargs)
		self.__bind(client)
		return client

	def __getattr__(self, name):
		return getattr(self.__sio, name)

	def __bind(self, sio):
		for event, handler in self.__handlers.items():
			sio.on(event, handler)

	@Logger.logged_method()
	def reset(self):
		self.__sio.disconnect()
		self.__sio = self.__init_sio()

	def on(self, event: str, handler: typing.Callable):
		self.__handlers[event] = handler
		self.__sio.on(event, handler)
