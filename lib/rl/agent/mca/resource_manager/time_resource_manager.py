import typing
from datetime import datetime

from lib.utils.logger import Logger
from .resource_manager import MCResourceManager


class TimeMCResourceManager(MCResourceManager):

	def __init__(self, step_time: float):
		self.__step_time = step_time
		Logger.info(f"Initialized {self.__class__.__name__} with step_time={step_time}")

	def init_resource(self) -> datetime:
		return datetime.now()

	def has_resource(self, resource: datetime) -> bool:
		return (datetime.now() - resource).seconds < self.__step_time
