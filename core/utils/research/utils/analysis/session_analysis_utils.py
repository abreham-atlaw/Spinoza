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
	def __extract_y_value_multi_channel(y: np.ndarray):
		return y[:, 0, -1]

	@staticmethod
	def __extract_y_value_legacy(y: np.ndarray):
		return y[:, -1]

	@staticmethod
	def get_timestep_pls(out_path: str, use_legacy: bool = False) -> np.ndarray:
		y = SessionAnalysisUtils.__load_y(out_path)
		values = SessionAnalysisUtils.__extract_y_value_multi_channel(y) if not use_legacy else SessionAnalysisUtils.__extract_y_value_legacy(y)
		assert values.ndim == 1
		pls = np.cumprod(values + 1)
		return pls
