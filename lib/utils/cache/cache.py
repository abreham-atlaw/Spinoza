from collections import deque


class Cache:

	def __init__(self, cache_size: int = 1000):
		self.__store = {}
		self.__order = deque()
		self.cache_size = cache_size

	@staticmethod
	def _hash(value) -> int:
		return hash(value)

	def store(self, key, value):
		hashed_key = self._hash(key)
		if hashed_key not in self.__store:
			# Check if we need to evict an item (FIFO)
			if len(self.__store) >= self.cache_size:
				oldest_key = self.__order.popleft()  # Remove the oldest inserted key
				self.__store.pop(oldest_key, None)

		self.__store[hashed_key] = value
		self.__order.append(hashed_key)  # Track the order of insertion

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
