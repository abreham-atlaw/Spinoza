import os.path

import numpy as np


class SessionAnalysisUtils:

	@staticmethod
	def __load_y(path: str) -> np.ndarray:
		y_path = os.path.join(path, "y")
		y = np.concatenate([
			np.load(os.path.join(y_path, filename))
			for filename in sorted(os.listdir(y_path))
		])
		return y

	@staticmethod
	def get_timestep_pls(out_path: str) -> np.ndarray:
		y = SessionAnalysisUtils.__load_y(out_path)
		values = y[:, -1]
		assert values.ndim == 1
		pls = np.cumprod(values + 1)
		return pls
