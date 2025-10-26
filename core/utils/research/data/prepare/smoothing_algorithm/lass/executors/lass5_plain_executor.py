import numpy as np

from .lass_executor import LassExecutor


class Lass5PlainExecutor(LassExecutor):

	def _execute(self, X: np.ndarray) -> np.ndarray:
		return self._model.predict(np.expand_dims(X, axis=0)).flatten()
