import json
import typing
from abc import ABC, abstractmethod
from datetime import datetime

from core import Config
from core.di import ServiceProvider
from core.utils.research.data.collect.runner_stats2 import RunnerStats2


class TimesRepository(ABC):

	@abstractmethod
	def retrieve_all(self) -> typing.List[datetime]:
		pass

	@abstractmethod
	def allocate(self, stat: RunnerStats2) -> datetime:
		pass


class JsonTimesRepository(TimesRepository):

	def __init__(
			self,
			path: str,
			format="%Y-%m-%d %H:%M:%S+00:00",
			resman_key: str = Config.TIMES_RESOURCE_MANAGER_KEY
	):
		self.__resource_manager = ServiceProvider.provide_resman(resman_key)
		self.__format = format
		with open(path, "r") as file:
			self.__times = map(
				lambda time_string: datetime.strptime(time_string, format),
				json.load(file)
			)

	def _generate_resource_key(self, stat: RunnerStats2, timestamp: str) -> str:
		return f"{stat.id}-{timestamp}"

	def retrieve_all(self) -> typing.List[datetime]:
		return self.__times

	def allocate(self, stat: RunnerStats2) -> datetime:

		for time in self.retrieve_all():
			resource_key = self._generate_resource_key(stat, time)
			if time.strftime(self.__format) not in stat.simulated_timestamps and (not self.__resource_manager.is_locked_by_id(resource_key)):
				self.__resource_manager.lock_by_id(resource_key)
				return time

		raise Exception(f"No available times for \"{stat.id}\".")

