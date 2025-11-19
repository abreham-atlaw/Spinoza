from abc import ABC, abstractmethod

import numpy as np

from core.utils.research.model.model.savable import SpinozaModule
from lib.rl.agent.dta import TorchModel


class LassExecutor(ABC):

	def __init__(self, model: SpinozaModule = None):
		self._model: TorchModel = None
		self._window_size: int = None
		if model is not None:
			self.set_model(model)

	def set_model(self, model: SpinozaModule):
		self._model = TorchModel(model)
		self._window_size = model.input_size[-1]

	@abstractmethod
	def _execute(self, X: np.ndarray) -> np.ndarray:
		pass

	def execute(self, X: np.ndarray) -> np.ndarray:
		return self._execute(X)

	@property
	def supports_batch_execution(self) -> bool:
		return False
