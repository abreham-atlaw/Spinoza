import typing
from abc import ABC, abstractmethod

import numpy as np

from core.agent.action import Action
from core.environment.trade_state import TradeState
from lib.rl.agent.dta import TorchModel


class StatePredictor(ABC):

	def __init__(self, model: TorchModel):
		self._model = model

	@abstractmethod
	def _prepare_input(self, states: typing.List[TradeState], actions: typing.List[Action], *args, **kwargs) -> np.ndarray:
		pass

	def _prepare_output(self, y: np.ndarray, states: typing.List[TradeState], actions: typing.List[Action], *args, **kwargs) -> np.ndarray:
		return y

	def predict(
			self,
			states: typing.List[TradeState],
			actions: typing.List[Action],
			*args, **kwargs
	) -> np.ndarray:
		X = self._prepare_input(states, actions, *args, **kwargs)
		y = self._model.predict(X)
		y = self._prepare_output(y, states, actions, *args, **kwargs)
		return y

	def __call__(self, *args, **kwargs):
		return self.predict(*args, **kwargs)
