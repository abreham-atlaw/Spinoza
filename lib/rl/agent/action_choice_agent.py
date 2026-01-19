from copy import deepcopy
from typing import *
from abc import ABC, abstractmethod

import numpy as np

import os
import json
import random


from lib.rl.environment import Environment
from .agent import Agent


CONFIGS_FILE_NAME = "configs.txt"


class ActionChoiceAgent(Agent):

	def __init__(self, explore_exploit_tradeoff: float = 0.3, update_agent=True):
		super().__init__()
		self.__environment: Union[Environment, None] = None
		self._explore_exploit_tradeoff = explore_exploit_tradeoff
		self._update_agent = update_agent

	def _validate_action(self, action, state) -> bool:
		return self._get_environment().is_action_valid(action, state)

	def _get_available_actions(self, state) -> List[object]:
		return [
			action
			for action in self._generate_actions(state)
			if self._validate_action(action, state)
		]

	@abstractmethod
	def _generate_actions(self, state) -> List[object]:
		pass

	@abstractmethod
	def _get_state_action_value(self, state, action, **kwargs) -> float:
		pass

	@abstractmethod
	def _update_state_action_value(self, initial_state, action, final_state, value):
		pass

	def _get_optimal_action(self, state, **kwargs):
		valid_actions = self._get_available_actions(state)

		values = [self._get_state_action_value(state, action, **kwargs) for action in valid_actions]

		optimal_action = valid_actions[values.index(max(values))]
		return optimal_action

	def _policy(self, state, **kwargs):
		return self._get_optimal_action(state, **kwargs)

	def _exploit(self, state):
		return self._policy(state)

	def _explore(self, state):
		actions = self._get_available_actions(state)
		return random.choice(actions)

	def _get_action(self, state):
		method = np.random.choice(
			[self._exploit, self._explore],
			1,
			p=[self._explore_exploit_tradeoff, 1-self._explore_exploit_tradeoff]
		)[0]
		return method(state)

	def perform_timestep(self):
		state = deepcopy(self._get_environment().get_state())
		action = self._get_action(state)
		value = self._get_environment().do(action)
		if self._update_agent:
			self._update_state_action_value(state, action, self._get_environment().get_state(), value)

	def get_configs(self) -> Dict:
		return {
			"explore_exploit_tradeoff": self._explore_exploit_tradeoff,
			"discount": self._discount_factor,
			"depth": self._depth,
		}

	def save(self, location):
		configs = self.get_configs()
		ActionChoiceAgent.save_configs(configs, location)

	@staticmethod
	def save_configs(configs: Dict, location):
		if os.path.isdir(location):
			os.mkdir(location)

		with open(os.path.join(location, CONFIGS_FILE_NAME), "w") as output:
			json.dump(configs, output)
			output.close()

	@staticmethod
	def load_configs(location) -> Dict:

		with open(os.path.join(location, CONFIGS_FILE_NAME)) as configs_file:
			data = json.load(configs_file)
			return data

	@classmethod
	def load_agent(cls, location):
		configs = cls.load_configs(location)
		return cls(**configs)
