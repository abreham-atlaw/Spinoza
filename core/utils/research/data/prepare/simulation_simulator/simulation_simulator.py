import typing
import pandas as pd
import numpy as np

from core.utils.research.data.prepare.smoothing_algorithm import SmoothingAlgorithm, MovingAverage
from core.utils.research.data.prepare.time_series_data_preparer import TimeSeriesDataPreparer
from core.utils.research.data.prepare.utils.data_prep_utils import DataPrepUtils
from lib.utils.logger import Logger


class SimulationSimulator(TimeSeriesDataPreparer):
	def __init__(
		self,
		df: pd.DataFrame,
		bounds: typing.List[float],
		seq_len: int,
		extra_len: int,
		batch_size: int,
		output_path: str,
		granularity: int,
		ma_window: int = None,
		order_gran: bool = True,
		smoothing_algorithm: typing.Optional[SmoothingAlgorithm] = None,
		**kwargs
	):
		if smoothing_algorithm is None and ma_window not in [None, 0, 1]:
			smoothing_algorithm = MovingAverage(ma_window)

		super().__init__(
			df=df,
			block_size=seq_len + (smoothing_algorithm.reduction if smoothing_algorithm is not None else 0) + 1,
			granularity=granularity,
			batch_size=batch_size,
			output_path=output_path,
			order_gran=order_gran,
			**kwargs
		)

		self.__bounds = bounds
		self.__extra_len = extra_len

		self._smoothing_algorithm = smoothing_algorithm
		Logger.info(f"Using Smoothing Algorithm: {self._smoothing_algorithm}")

	def _prepare_sequence_stack(self, x: np.ndarray) -> np.ndarray:
		if self._smoothing_algorithm is not None:
			x = self._smoothing_algorithm.apply_on_batch(x)
		return x

	def _prepare_x(self, sequences: np.ndarray) -> np.ndarray:
		return np.concatenate(
			(
				sequences[:, :-1],
				np.zeros((sequences.shape[0], self.__extra_len))
			),
			axis=1
		)

	def _prepare_y(self, sequences: np.ndarray) -> np.ndarray:
		percentages = sequences[:, -1] / sequences[:, -2]
		classes = np.array([self.__find_gap_index(p) for p in percentages])
		encoding = self.__one_hot_encode(classes, len(self.__bounds) + 1)
		return np.concatenate(
			(
				encoding,
				np.zeros((encoding.shape[0], 1))
			),
			axis=1
		)

	def __find_gap_index(self, number: float) -> int:
		for i, bound in enumerate(self.__bounds):
			if number < bound:
				return i
		return len(self.__bounds)

	@staticmethod
	def __one_hot_encode(classes: np.ndarray, length: int) -> np.ndarray:
		encoding = np.zeros((classes.shape[0], length))
		for i in range(classes.shape[0]):
			encoding[i, classes[i]] = 1
		return encoding

	def _batch_df(self, df: pd.DataFrame) -> typing.List[pd.DataFrame]:
		instruments = DataPrepUtils.get_instruments(df)
		Logger.info(f"Found {len(instruments)} instruments: {instruments}")
		if len(instruments) == 1:
			return super()._batch_df(df)

		Logger.info(f"Using instrument batching...")
		dfs = [
			df[(df["base_currency"] == base_currency) & (df["quote_currency"] == quote_currency)]
			for base_currency, quote_currency in instruments
		]

		Logger.info("Instrument Sizes: \n {}".format("\n".join([f"{base_currency}-{quote_currency}: {i_df.shape[0]}" for (base_currency, quote_currency), i_df in zip(instruments, dfs)])))

		return dfs
