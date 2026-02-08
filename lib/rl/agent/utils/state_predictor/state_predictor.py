import typing
from abc import ABC, abstractmethod

import numpy as np

from lib.rl.agent.dta import TorchModel
from lib.rl.environment import ModelBasedState


class StatePredictor(ABC):

	def __init__(self, model: TorchModel):
		self._model = model

	@abstractmethod
	def prepare_input(self, states: typing.List[ModelBasedState], actions: typing.List[typing.Any], *args, **kwargs) -> np.ndarray:
		pass

	def _prepare_output(self, y: np.ndarray, states: typing.List[ModelBasedState], actions: typing.List[typing.Any], *args, **kwargs) -> np.ndarray:
		return y

	def predict(
			self,
			states: typing.List[ModelBasedState],
			actions: typing.List[typing.Any],
			*args, **kwargs
	) -> np.ndarray:
		X = self.prepare_input(states, actions, *args, **kwargs)
		y = self._model.predict(X)
		y = self._prepare_output(y, states, actions, *args, **kwargs)
		return y

	def __call__(self, *args, **kwargs):
		return self.predict(*args, **kwargs)
