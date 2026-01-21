import time
from typing import *
from abc import ABC, abstractmethod

import socketio

from lib.rl.agent import MonteCarloAgent
from lib.network.rest_interface import Serializer


class MonteCarloQueen(MonteCarloAgent, ABC):

	def __init__(self, server_url: str, *args, wait_time=0.01, **kwargs):
		super().__init__(*args, **kwargs)
		self.__socketio = socketio.Client(logger=True)
		self.__socketio.connect(server_url)
		self.__map_events()
		self.__state_serializer = self._init_state_serializer()
		self.__action_serializer = self._init_action_serializer()
		self.__wait_time = wait_time
		self.__reset_received_action()

	@abstractmethod
	def _init_state_serializer(self) -> Serializer:
		pass

	@abstractmethod
	def _init_action_serializer(self) -> Serializer:
		pass

	def _map_events(self) -> List[Tuple[str, object]]:
		return [
			("action", self.__handle_action),
		]

	def __map_events(self):
		for event, handler in self._map_events():
			self.__socketio.on(event, handler=handler)

	def __reset_received_action(self):
		self.__optimal_action, self.__action_received = None, False

	def __get_received_action(self):
		return self.__optimal_action

	def __is_action_received(self) -> bool:
		return self.__action_received

	def __set_received_action(self, action):
		self.__optimal_action = action
		self.__action_received = True

	def __handle_action(self, action=None):
		self.__set_received_action(self.__action_serializer.deserialize(action))

	def _monte_carlo_tree_search(self, state) -> object:
		print("Performing MonteCarlo Tree Search")
		self.__socketio.emit("new", self.__state_serializer.serialize(state))
		self.__reset_received_action()

		resources = self._init_resources()

		while self._has_resource(resources):
			time.sleep(self.__wait_time)

		self.__socketio.emit("end")
		while not self.__is_action_received():
			time.sleep(self.__wait_time)

		return self.__get_received_action()

	def _get_optimal_action(self, state, **kwargs):
		self._monte_carlo_tree_search(state)
		return self.__get_received_action()