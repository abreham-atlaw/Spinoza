import functools

import numpy as np
import pandas as pd

from lib.utils.logger import Logger
from .simulation_simulator3 import SimulationSimulator3
from ..utils.data_prep_utils import DataPrepUtils


class SimulationSimulator4(SimulationSimulator3):

	def __init__(self, *args, **kwargs):
		super().__init__(
			*args,
			allow_instrument_batching=False,
			**kwargs
		)

	@staticmethod
	def __split_instruments_dfs(df) -> pd.DataFrame:
		instruments = DataPrepUtils.get_instruments(df)
		dfs = [
			df[(df["base_currency"] == base_currency) & (df["quote_currency"] == quote_currency)]
			for base_currency, quote_currency in instruments
		]
		return dfs

	@staticmethod
	def __extract_common_times(df: pd.DataFrame) -> pd.DataFrame:
		Logger.info(f"Extracting common times...")
		dfs = SimulationSimulator4.__split_instruments_dfs(df)

		common_times = functools.reduce(
			lambda a, b: a.intersection(b),
			[set(ins_df["time"]) for ins_df in dfs]
		)

		dfs = [
			ins_df[ins_df["time"].isin(common_times)].copy()
			for ins_df in dfs
		]

		new_df = pd.concat(dfs)

		return new_df

	def _setup_df(self, df: pd.DataFrame) -> pd.DataFrame:
		df = self.__extract_common_times(df)
		return super()._setup_df(df)

	def _extract_columns(self, df: pd.DataFrame) -> np.ndarray:
		dfs = self.__split_instruments_dfs(df)
		data = np.concatenate([
			super(SimulationSimulator4, self)._extract_columns(df)
			for df in dfs
		], axis=0)
		return data

	def _prepare_y(self, sequences: np.ndarray) -> np.ndarray:
		return np.concatenate([
			super(SimulationSimulator4, self)._prepare_y(sequences[:, i*len(self._x_columns): (i+1)*len(self._x_columns)])
			for i in range(sequences.shape[1] // len(self._x_columns))
		], axis=1)

	def _extract_granularity(self, data: np.ndarray, g: int) -> np.ndarray:
		return np.concatenate([
			super(SimulationSimulator4, self)._extract_granularity(
				data=data[i*len(self._x_columns): (i+1)*len(self._x_columns)],
				g=g
			)
			for i in range(data.shape[0] // len(self._x_columns))
		], axis=0)

	def _prepare_sequence_stack(self, x: np.ndarray) -> np.ndarray:
		return np.concatenate([
			super(SimulationSimulator4, self)._prepare_sequence_stack(
				x[:, i*len(self._x_columns): (i+1)*len(self._x_columns)]
			)
			for i in range(x.shape[1] // len(self._x_columns))
		], axis=1)