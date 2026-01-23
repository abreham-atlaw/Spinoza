import typing
from abc import ABC, abstractmethod

from lib.rl.environment import Environment
from lib.utils.logger import Logger


class Agent(ABC):

	def __init__(self):
		pass

	def _get_environment(self) -> Environment:
		if self.__environment is None:
			raise Exception("Environment Not Set.")
		return self.__environment

	def set_environment(self, environment: Environment):
		self.__environment = environment

	def _is_episode_over(self, state) -> bool:
		return self._get_environment().is_episode_over(state)

	@abstractmethod
	def _policy(self, state) -> typing.Any:
		pass

	def perform_timestep(self):
		state = self._get_environment().get_state()
		action = self._policy(state)
		return self._get_environment().do(action)

	def perform_episode(self):
		while not self._get_environment().is_episode_over():
			self.perform_timestep()
		Logger.success("Episode over. Exiting...")

	def loop(self):
		if self._get_environment().is_episodic():
			while True:
				self.perform_episode()
				self._get_environment().reset()
		else:
			while True:
				self.perform_timestep()