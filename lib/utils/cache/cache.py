import typing
from collections import deque


class Cache:

	def __init__(self, cache_size: int = 1000, key_func: typing.Callable = None):
		self.__store = {}
		self.__order = deque()
		self.cache_size = cache_size
		self.__key_func = key_func if key_func is not None else lambda x: x

	def _hash(self, value) -> int:
		return hash(self.__key_func(value))

	def store(self, key, value):
		hashed_key = self._hash(key)
		if hashed_key not in self.__store:
			if len(self.__store) >= self.cache_size:
				oldest_key = self.__order.popleft()
				self.__store.pop(oldest_key, None)

		self.__store[hashed_key] = value
		self.__order.append(hashed_key)

	def retrieve(self, key):
		return self.__store.get(self._hash(key))

	def cached_or_execute(self, key, func):
		value = self.retrieve(key)
		if value is None:
			value = func()
			self.store(key, value)
		return value

	def remove(self, key):
		hashed_key = self._hash(key)
		if hashed_key in self.__store:
			self.__store.pop(hashed_key, None)
			self.__order.remove(hashed_key)  # Remove the key from the order tracking

	def clear(self):
		self.__store = {}
		self.__order.clear()  # Clear the order tracking as well

	def __setitem__(self, key, value):
		self.store(key, value)

	def __getitem__(self, key):
		return self.retrieve(key)
