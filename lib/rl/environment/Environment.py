from typing import *
from abc import ABC, abstractmethod

from lib.utils.logger import Logger


class Environment(ABC):

	def __init__(self, episodic=True):
		self.episodic = episodic

	@abstractmethod
	def get_reward(self, state=None) -> float:
		pass

	@abstractmethod
	def perform_action(self, action):
		pass

	@abstractmethod
	def render(self):
		pass

	@abstractmethod
	def update_ui(self):
		pass

	@abstractmethod
	def check_is_running(self) -> bool:
		pass

	@abstractmethod
	def is_action_valid(self, action, state) -> bool:
		pass

	@abstractmethod
	def get_state(self):
		pass

	@abstractmethod
	def is_episode_over(self, state=None) -> bool:
		pass

	@Logger.logged_method
	def do(self, action) -> float:
		if not self.is_action_valid(action, self.get_state()):
			raise ActionNotValidException()
		self.perform_action(action)
		self.update_ui()
		return self.get_reward()

	def is_episodic(self) -> bool:
		return self.episodic

	def reset(self):
		self.start()

	def _initialize(self):
		self.render()

	def get_latest_state(self):
		return self.get_state()

	def start(self):
		self._initialize()


class ActionNotValidException(Exception):
	pass
