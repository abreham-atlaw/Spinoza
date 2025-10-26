import numpy as np

from .lass_executor import LassExecutor


class Lass2Executor(LassExecutor):

	def __construct_input(self, x: np.ndarray, y: np.ndarray, i: int) -> np.ndarray:
		inputs = np.zeros((1, 2, self._window_size))
		inputs[0, 0, :] = x[i: i + self._window_size]
		inputs[0, 1, -(i+1):-1] = y[max(0, i-(self._window_size-1)):i]
		return inputs

	def _execute(self, X: np.ndarray) -> np.ndarray:

		y = np.zeros(X.shape[0] - self._window_size + 1)

		for i in range(y.shape[0]):
			prediction = self._model.predict(self.__construct_input(X, y, i))
			y[i] = prediction.flatten()[0]

		return y
