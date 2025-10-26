import typing

import numpy as np

from core.utils.research.data.prepare.smoothing_algorithm import SmoothingAlgorithm
from core.utils.research.data.prepare.smoothing_algorithm.lass.executors import BasicLassExecutor
from core.utils.research.data.prepare.smoothing_algorithm.lass.executors.lass_executor import LassExecutor
from core.utils.research.data.prepare.utils.data_prep_utils import DataPrepUtils
from core.utils.research.model.model.savable import SpinozaModule
from lib.rl.agent.dta import TorchModel
from lib.utils.torch_utils.model_handler import ModelHandler


class Lass(SmoothingAlgorithm):

	def __init__(
			self,
			model: typing.Union[SpinozaModule, str],
			executor: LassExecutor = None
	):
		super().__init__()
		if executor is None:
			executor = BasicLassExecutor(model)
		self.__executor = executor
		self.__executor.set_model(model)

	@property
	def reduction(self) -> int:
		return 0

	def __apply(self, x: np.ndarray) -> np.ndarray:
		return self.__executor.execute(x)

	def apply(self, x: np.ndarray) -> np.ndarray:
		return self.__apply(x)

	def apply_on_batch(self, x: np.ndarray) -> np.ndarray:
		if self.__executor.supports_batch_execution:
			return self.__apply(x)
		return super().apply_on_batch(x)
