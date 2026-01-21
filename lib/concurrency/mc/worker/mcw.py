import time
from typing import *
from abc import ABC, abstractmethod

import socketio

from lib.rl.agent import MonteCarloAgent
from lib.concurrency.mc.data.staterepository import DistributedStateRepository, SocketIOChannel
from lib.network.rest_interface.serializers import Serializer
from lib.utils.logger import Logger


class MonteCarloWorkerAgent(MonteCarloAgent, ABC):

	def __init__(self, server_url, *args, **kwargs):
		super().__init__(*args, **kwargs)
		self.__socketio = self._init_socket_io()
		self.__state_serializer = self._init_state_serializer()
		self.__graph_serializer = self._init_graph_serializer()
		self.__active = False
		self.__url = server_url

	@abstractmethod
	def _init_socket_io(self) -> socketio.Client:
		pass

	@abstractmethod
	def _init_state_serializer(self) -> Serializer:
		pass

	@abstractmethod
	def _init_graph_serializer(self) -> Serializer:
		pass

	def _map_events(self) -> List[Tuple[str, object]]:
		return [
			("new", self.__handle_new),
			("select", self.__handle_select),
			("end", self.__handle_end),
		]

	def __map_events(self):
		for event, handler in self._map_events():
			self.__socketio.on(event, handler=handler)

	def _set_active(self, active):
		self.__active = active
		if active:
			self.__socketio.emit("select")
		if not active:
			self._state_repository.clear()

	def _is_active(self):
		return self.__active

	def __handle_new(self):
		print("Got New")
		self._set_active(True)

	def __handle_select(self, node):
		node = self.__graph_serializer.deserialize_json(node)
		Logger.info("Working on %s " % (node.id,))
		self._monte_carlo_simulation(node)
		json = self.__graph_serializer.serialize_json(node)
		self.__socketio.emit("backpropagate", json)
		if self._is_active():
			self.__socketio.emit("select")
			Logger.info("Waiting for select")

	def __handle_end(self):
		self._set_active(False)

	def start(self):
		self.__socketio.connect(self.__url)
		self.__map_events()
		self._set_active(True)
		self.__socketio.wait()

