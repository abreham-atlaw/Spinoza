import numpy as np

from core.utils.research.data.prepare.utils.data_prep_utils import DataPrepUtils
from core.utils.research.model.model.savable import SpinozaModule
from .lass_executor import LassExecutor


class BasicLassExecutor(LassExecutor):

	def __init__(self, model: SpinozaModule = None):
		super().__init__(model)
		self.__window_size = None

	def _execute(self, X: np.ndarray) -> np.ndarray:
		x = DataPrepUtils.stack(X, self._window_size)
		y = self._model.predict(x).flatten()
		return y
