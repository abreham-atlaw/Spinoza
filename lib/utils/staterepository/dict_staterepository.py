import typing

from lib.network.rest_interface import Serializer
from .staterepository import StateRepository, StateNotFoundException


class DictStateRepository(StateRepository):

	def __init__(self):
		super().__init__()
		self.__states = {}

	def store(self, key: str, state):
		self.__states[key] = state

	def retrieve(self, key: str) -> object:
		try:
			return self.__states[key]
		except KeyError:
			raise StateNotFoundException()

	def exists(self, key: str) -> bool:
		return key in self.__states.keys()

	def clear(self):
		self.__states = {}

	def destroy(self):
		del self.__states

	def get_keys(self) -> typing.List[str]:
		return list(self.__states.keys())

class SectionalDictStateRepository(StateRepository):

	def __init__(self, len_: int, depth: int = 1, serializer: Serializer = None):
		super().__init__(serializer)
		self.__len = len_
		self.__depth = depth
		self.__states = {}
		self.__keys = set()

	def __store(self, key: str, state, dict_: dict, depth: int):
		if depth == 0:
			dict_[key] = state
			return
		inner_key = key[:self.__len]
		inner_dict = dict_.get(inner_key)
		if inner_dict is None:
			inner_dict = {}
			dict_[inner_key] = inner_dict

		self.__store(
			key[self.__len:],
			state,
			inner_dict,
			depth-1
		)

	def __retrieve(self, key: str, dict_: dict, depth: int) -> object:
		if depth == 0:
			return dict_[key]
		return self.__retrieve(key[self.__len:], dict_[key[:self.__len]], depth-1)

	def store(self, key: str, state):
		if len(key) <= self.__len * self.__depth:
			raise Exception("Key Too Short. Must be more than %s characters long" % (self.__len * self.__depth, ))
		self.__store(key, state, self.__states, self.__depth)
		self.__keys.add(key)

	def retrieve(self, key: str) -> object:
		try:
			value = self.__retrieve(key, self.__states, self.__depth)
			if value is None:
				raise StateNotFoundException()
			return value
		except KeyError:
			raise StateNotFoundException

	def exists(self, key: str) -> bool:
		try:
			self.retrieve(key)
			return True
		except StateNotFoundException:
			return False

	def clear(self):
		self.__states = {}
		self.__keys = set()

	def destroy(self):
		del self.__states
		self.__keys = set()

	def get_keys(self) -> typing.List[str]:
		return list(self.__keys)
