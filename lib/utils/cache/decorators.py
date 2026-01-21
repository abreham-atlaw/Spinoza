import typing
from datetime import datetime

from .cache import Cache


class CacheDecorators:

	__caches_store = {}

	@staticmethod
	def __get_method_cache(instance, method, size):
		attribute_name = f"{method.__name__}__cache"
		if not hasattr(instance, attribute_name):
			setattr(instance, attribute_name, Cache(cache_size=size))
		return getattr(instance, attribute_name)

	@staticmethod
	def __get_cache_keys(args, kwargs, timeout=None) -> str:
		if timeout is None:
			time_key = "null",
		else:
			now = datetime.now()
			time_key = str(now.replace(minute=(now.minute//timeout)*timeout, second=0, microsecond=0).timestamp())
			print(f"Time Key: {time_key}(Timeout={timeout}, now={now})")
		return str({
			"args": [str(a) for a in args],
			"kwargs": {
				key: str(kwargs.get(key))
				for key in kwargs
			},
			"time": time_key
		})

	@staticmethod
	def cached_method(timeout=None, size=1000):
		def decorator(func):
			def wrapper(self, *args, **kwargs):
				cache = CacheDecorators.__get_method_cache(self, func, size=size)
				return cache.cached_or_execute(
					CacheDecorators.__get_cache_keys(args, kwargs, timeout=timeout),
					func=lambda: func(self, *args, **kwargs)
				)
			return wrapper
		return decorator

	@staticmethod
	def __get_function_cache(func: typing.Callable) -> Cache:
		cache = CacheDecorators.__caches_store.get(func)
		if cache is None:
			cache = Cache()
			CacheDecorators.__caches_store[func] = cache
		return cache

	@staticmethod
	def singleton(timeout: int = None):
		def decorator(func):
			def wrapper(*args, **kwargs):
				cache = CacheDecorators.__get_function_cache(func)
				return cache.cached_or_execute(
					CacheDecorators.__get_cache_keys(args, kwargs, timeout=timeout),
					func=lambda: func(*args, **kwargs)
				)
			return wrapper
		return decorator