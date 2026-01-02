import numpy as np

from core.utils.research.model.model.savable import SpinozaModule
from .lass_executor import LassExecutor

class Lass8Executor(LassExecutor):

	@property
	def supports_batch_execution(self) -> bool:
		return False

	def _execute(self, X: np.ndarray) -> np.ndarray:
		assert X.ndim == 1
		if (X.shape[0] % self._window_size) != 0:
			raise Exception(f"Input Size({X.shape[0]}) must be a multiple of window size({self._window_size})")

		x = np.reshape(X, (-1, self._window_size))
		y = self._model.predict(x)
		return y.flatten()
