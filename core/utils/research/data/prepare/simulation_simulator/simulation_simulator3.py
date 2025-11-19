import typing

import numpy as np
import pandas as pd
import torch

from lib.utils.logger import Logger
from .simulation_simulator import SimulationSimulator
from ..utils.data_prep_utils import DataPrepUtils


class SimulationSimulator3(SimulationSimulator):

	def __init__(
			self,
			*args,
			x_columns: typing.List[str] = (
					"c", "o", "l", "h", "v",
					"time.year", "time.month", "time.day", "time.hour", "time.minute", "time.second"
			),
			smoothed_columns: typing.Tuple[str, ...] = ("c", "o", "l", "h"),
			y_columns: typing.List[str] = ("c",),
			flatten_y: bool = True,
			**kwargs
	):
		super().__init__(*args, **kwargs)
		Logger.info(f"Using X Columns: {x_columns}")
		Logger.info(f"Using Smoothed Columns: {smoothed_columns}")
		Logger.info(f"Using Y Columns: {y_columns}")
		self.__x_columns = x_columns
		self.__smoothed_columns = smoothed_columns
		self.__y_columns = y_columns
		self.__flatten_y = flatten_y

	@staticmethod
	def _encode_timestamp(df: pd.DataFrame) -> pd.DataFrame:
		timestamp = pd.to_datetime(df["time"].to_numpy())
		df[["time.year", "time.month", "time.day", "time.hour", "time.minute", "time.second"]] = np.stack(
			[timestamp.year, timestamp.month, timestamp.day, timestamp.hour, timestamp.minute, timestamp.second],
			axis=1)
		return df

	def _setup_df(self, df: pd.DataFrame) -> pd.DataFrame:
		df = super()._setup_df(df)
		df = self._encode_timestamp(df.copy())
		return df

	def _extract_columns(self, df: pd.DataFrame) -> np.ndarray:
		return df[list(self.__x_columns)].to_numpy().transpose((1, 0))

	def _prepare_sequence_stack(self, x: np.ndarray) -> np.ndarray:
		if self._smoothing_algorithm is None:
			return x

		x = np.stack([
			self._smoothing_algorithm.apply_on_batch(x[:, i, :])
			if self.__x_columns[i] in self.__smoothed_columns
			else x[:, i, self._smoothing_algorithm.reduction:]
			for i in range(x.shape[1])
		], axis=1)

		return x

	def _prepare_x(self, sequences: np.ndarray) -> np.ndarray:
		return np.stack([
			super(SimulationSimulator3, self)._prepare_x(sequences[:, i, :])
			for i in range(sequences.shape[1])
		], axis=1)

	def _prepare_y(self, sequences: np.ndarray) -> np.ndarray:
		y = np.stack([
			super(SimulationSimulator3, self)._prepare_y(sequences[:, i, :])
			for i in range(sequences.shape[1])
			if self.__x_columns[i] in self.__y_columns
		], axis=1)

		if y.shape[1] == 1 and self.__flatten_y:
			y = np.squeeze(y, axis=1)

		return y

	def _extract_granularity(self, data: np.ndarray, g: int) -> np.ndarray:
		df = pd.DataFrame(columns=self.__x_columns, data=data.transpose((1, 0)))
		return self._extract_columns(DataPrepUtils.condense_granularity(df, g))
