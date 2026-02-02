import time
import typing

from lib.utils.logger import Logger


def handle_exception(exception_cls=(Exception,), callback: typing.Callable[[Exception], None]=None):
	def decorator(func):
		def wrapper(*args, **kwargs):
			exceptions = tuple(exception_cls) if isinstance(exception_cls, (list, tuple)) else (exception_cls,)
			try:
				return func(*args, **kwargs)
			except exceptions as ex:
				Logger.warning(f"Exception Raised: {ex.__class__.__name__} while calling {func.__name__}. {ex}")
				if callback is not None:
					callback(ex)
		return wrapper
	return decorator
