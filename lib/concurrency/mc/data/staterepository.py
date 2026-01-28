import typing
from abc import ABC, abstractmethod

from flask_socketio import emit
import socketio
import json
from datetime import datetime
import time

from lib.utils.staterepository import PickleStateRepository, DictStateRepository
from lib.network.rest_interface.serializers import Serializer
from lib.utils.staterepository.staterepository import StateNotFoundException


class Channel(ABC):

	@abstractmethod
	def emit(self, event, *args, **kwargs):
		pass

	@abstractmethod
	def map(self, event, handler):
		pass


class SocketIOChannel(Channel):

	def __init__(self, socketio):
		self._socketio = socketio

	def _emit(self, *args, **kwargs):
		self._socketio.emit(*args, **kwargs)

	def emit(self, event, *args, **kwargs):
		kwargs = {key: value for key, value in kwargs.items() if value}
		self._emit(event, *args, **kwargs)

	def map(self, event, handler):
		self._socketio.on(event, handler)


class FlaskSocketIOChannel(SocketIOChannel):

	def _emit(self, *args, **kwargs):
		emit(*args, **kwargs)

	def map(self, event, handler):
		self._socketio.on_event(event, handler)


class DistributedStateRepository(DictStateRepository):

	def __init__(
			self,
			channel: Channel,
			serializer: Serializer,
			*args,
			timeout=3,
			is_server=False,
			**kwargs
	):
		super().__init__(*args, **kwargs)
		self.__serializer = serializer
		self.__channel = channel
		self.__timeout = timeout
		self.__is_server = is_server
		self.__map_events()

	def __map_events(self):
		self.__channel.map("state_response", self.__handle_response)
		self.__channel.map("state_request", self.__handle_request)

	def __handle_response(self, response):
		self.store(response["id"], self.__serializer.deserialize(response["state"]))

	def __handle_request(self, key):
		try:
			state = self.retrieve(key, broadcast=self.__is_server)
		except StateNotFoundException:
			return
		self.__channel.emit(
			"state_response",
			{
				"id": key,
				"state": self.__serializer.serialize(state)
			}
		)

	def __wait_retrieve(self, key) -> object:
		start_time = datetime.now()
		value = None
		while value is None and ((datetime.now() - start_time).total_seconds() < self.__timeout or True):
			try:
				value = super().retrieve(key)
			except StateNotFoundException:
				time.sleep(self.__timeout/100)
		return value

	def retrieve(self, key: str, broadcast=True) -> object:
		try:
			value = super().retrieve(key)
		except StateNotFoundException:
			if broadcast:
				self.__channel.emit("state_request", key, broadcast=self.__is_server)
				value = self.__wait_retrieve(key)
			else:
				raise StateNotFoundException()
		return value

	def get_keys(self) -> typing.List[str]:
		pass
