import numpy as np

from .transformation import Transformation


class TimeStretchTransformation(Transformation):

	def __init__(self, min_n: float = 0.75, max_n: float = 1):
		super().__init__()
		self.min_n = min_n
		self.max_n = max_n

	def _transform(self, x: np.ndarray) -> np.ndarray:
		n_hat = np.random.randint(np.floor(self.min_n * x.shape[1]), np.ceil(self.max_n * x.shape[1]), (x.shape[0], 1))
		n = x.shape[1]
		r = (n - 1)/(n_hat - 1)
		d = n - n_hat
		i_hat = np.repeat(np.expand_dims(np.arange(n), axis=0), axis=0, repeats=x.shape[0])
		t = np.floor(i_hat * 1 / r).astype(np.int32)
		i = (t + d).astype(np.int32)
		x_hat = x[np.arange(x.shape[0]).reshape((-1, 1)), i] + \
				(i_hat - t * r) * (
						(
								np.concatenate((x, np.zeros((x.shape[0], 1))), axis=1)[np.arange(x.shape[0]).reshape((-1, 1)), i + 1] -
								x[np.arange(x.shape[0]).reshape((-1, 1)), i]
						) / r
				)
		return x_hat
