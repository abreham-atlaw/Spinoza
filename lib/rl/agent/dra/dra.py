import typing
from abc import ABC, abstractmethod

import numpy as np

from lib.rl.agent import ActionChoiceAgent
from lib.rl.agent.dra.generator import AgentDataGenerator
from lib.rl.agent.dta import Model
from lib.rl.agent.utils.state_predictor import StatePredictor


class DeepReinforcementAgent(ActionChoiceAgent, ABC):

	def __init__(
			self,
			*args,
			batch_size=32,
			train: bool = False,
			save_path: typing.Optional[str] = None,
			**kwargs
	):
		super().__init__(*args, **kwargs)
		self._predictor = self._init_predictor()
		self.__generator = AgentDataGenerator(batch_size, export_path=save_path)
		self.__save, self.__train = save_path is not None, train

	@abstractmethod
	def _init_predictor(self) -> StatePredictor:
		pass

	@abstractmethod
	def _prepare_dra_output(self, state: typing.Any, action: typing.Any, output: np.ndarray) -> float:
		pass

	@abstractmethod
	def _prepare_dra_train_output(self, state: typing.Any, action: typing.Any, final_state: typing.Any, value: float) -> np.ndarray:
		pass

	def _get_state_action_value(self, state, action, **kwargs) -> float:
		return self._prepare_dra_output(
			state,
			action,
			np.squeeze(
				self._predictor.predict(
					[state],
					[action]
				),
			)
		)

	def _update_state_action_value(self, initial_state, action, final_state, value):
		self.__generator.append(
			self._prepare_dra_input(initial_state, action),
			self._prepare_dra_train_output(initial_state, action, final_state, value)
		)
		if len(self.__generator) == 2:
			X, y = self.__generator[0]
			if self.__save:
				self.__generator.save()
			self.__generator.remove(0)
