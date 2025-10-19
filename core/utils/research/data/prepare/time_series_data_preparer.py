import typing
from abc import ABC, abstractmethod

import numpy as np
import pandas as pd

import os
from datetime import datetime

from core.utils.research.data.prepare.augmentation import Transformation
from core.utils.research.data.prepare.splitting import TrainTestSplitter
from core.utils.research.data.prepare.utils.data_prep_utils import DataPrepUtils
from lib.utils.logger import Logger


class TimeSeriesDataPreparer(ABC):

	def __init__(
			self,
			df: pd.DataFrame,
			block_size: int,
			granularity: int,
			batch_size: int,
			output_path: str,
			order_gran: bool = True,
			trim_extra_gran: bool = False,
			process_batch_size: int = None,
			trim_incomplete_batch: bool = False,
			clean_df: bool = True,

			X_dir: str = "X",
			y_dir: str = "y",
			train_dir: str = "train",
			test_dir: str = "test",
			splitter: TrainTestSplitter = None,

			transformations: typing.List[Transformation] = None
	):
		if clean_df:
			df = DataPrepUtils.clean_df(df)

		self.__df = df
		self.__block_size = block_size
		self.__granularity = granularity
		self.__batch_size = batch_size
		self.__output_path = output_path
		self.__order_gran = order_gran
		self.__trim_extra_gran = trim_extra_gran
		self.__process_batch_size = process_batch_size
		self.__trim_incomplete_batch = trim_incomplete_batch

		self.__X_dir, self.__y_dir = X_dir, y_dir
		self.__train_dir, self.__test_dir = train_dir, test_dir
		self.__splitter = splitter
		Logger.info(f"Using splitter: {self.__splitter}")

		if transformations is None:
			transformations = []
		self.__transformations = transformations
		Logger.info(f"Using transformations({len(self.__transformations)}): {', '.join(map(lambda t: str(t), self.__transformations))}")

	@staticmethod
	def __generate_filename() -> np.ndarray:
		return f"{datetime.now().timestamp()}.npy"

	def __filter_df(self, start_date: datetime = None, end_date: datetime = None) -> pd.DataFrame:
		df = self.__df
		if start_date is not None:
			df = df[df["time"] >= start_date]
		if end_date is not None:
			df = df[df["time"] <= end_date]
		return df

	@abstractmethod
	def _prepare_x(self, sequences: np.ndarray) -> np.ndarray:
		pass

	@abstractmethod
	def _prepare_y(self, sequences: np.ndarray) -> np.ndarray:
		pass

	def _prepare_sequence(self, sequence: np.ndarray) -> np.ndarray:
		return sequence

	def __apply_transformations(self, x: np.ndarray) -> np.ndarray:
		y = np.concatenate([x] + [
			transformation(x)
			for transformation in self.__transformations
		])
		return y

	def _prepare_sequence_stack(self, x: np.ndarray) -> np.ndarray:
		return x

	def __process_sequence(self, sequence: np.ndarray) -> typing.Tuple[np.ndarray, np.ndarray]:
		stacked_sequence = DataPrepUtils.stack(sequence, self.__block_size)

		stacked_sequence = self.__apply_transformations(stacked_sequence)

		stacked_sequence = self._prepare_sequence_stack(stacked_sequence)

		X = self._prepare_x(stacked_sequence)
		y = self._prepare_y(stacked_sequence)

		return X, y

	def __trim_grans(self, arrays: typing.List[np.ndarray]) -> np.ndarray:
		length = min(arr.shape[0] for arr in arrays)
		return [arr[:length] for arr in arrays]

	def __concatenate_grans(self, arrays: typing.List[np.ndarray]) -> np.ndarray:

		def p(g, i, G):
			return g + G*i

		if not self.__order_gran:
			return np.concatenate(arrays, axis=0)

		if self.__trim_extra_gran:
			arrays = self.__trim_grans(arrays)

		new_arr = np.zeros((sum(arr.shape[0] for arr in arrays), *arrays[0].shape[1:]))
		for i in range(len(arrays)):
			new_arr[p(i, np.arange(arrays[i].shape[0]), len(arrays))] = arrays[i]
		return new_arr

	def __prepare_data(self, df: pd.DataFrame) -> typing.Tuple[np.ndarray, np.ndarray]:
		Logger.info(f"Preparing Data...")

		Xs, ys = [], []

		for i in range(self.__granularity):
			gran_sequence = df["c"].to_numpy()[i::self.__granularity]
			gran_sequence = self._prepare_sequence(gran_sequence)
			gran_X, gran_y = self.__process_sequence(gran_sequence)

			Xs.append(gran_X)
			ys.append(gran_y)

		X, y = [self.__concatenate_grans(arrays) for arrays in [Xs, ys]]
		return X, y

	def __batch_array(
			self,
			X: np.ndarray,
			y: np.ndarray
	) -> typing.Tuple[typing.List[np.ndarray], typing.List[np.ndarray]]:
		Logger.info(f"Batching Array...")
		size = X.shape[0]

		X_batches = []
		y_batches = []

		for i in range(0, size, self.__batch_size):
			X_batches.append(X[i: i + self.__batch_size])
			y_batches.append(y[i: i + self.__batch_size])

		if self.__trim_incomplete_batch:
			X_batches, y_batches = [
				list(filter(
					lambda batch: batch.shape[0] == self.__batch_size,
					batches
				))
				for batches in (X_batches, y_batches)
			]

		return X_batches, y_batches

	def __save(self, X: np.ndarray, y: np.ndarray, purpose_dir: str):
		filename = self.__generate_filename()

		for arr, path in zip((X, y), (self.__X_dir, self.__y_dir)):
			save_path = os.path.join(self.__output_path, purpose_dir, path, filename)
			if not os.path.exists(os.path.dirname(save_path)):
				os.makedirs(os.path.dirname(save_path))
			np.save(save_path, arr)

	def __save_batches(self, X_batches, y_batches, purpose_dir: str):
		Logger.info(f"Saving {purpose_dir} Batches to {self.__output_path}...")
		for i, (X, y) in enumerate(zip(X_batches, y_batches)):
			self.__save(X, y, purpose_dir)
			Logger.info(f"{(i+1) * 100 / len(X_batches) :.2f}", end="\r")

	def __batch_and_save(self, X: np.ndarray, y: np.ndarray, purpose_dir: str):
		Logger.info(f"Processing {purpose_dir}...")
		X_batches, y_batches = self.__batch_array(X, y)
		self.__save_batches(X_batches, y_batches, purpose_dir)

	def __split(self, X: np.ndarray, y:np.ndarray) -> typing.Tuple[typing.Tuple[np.ndarray, np.ndarray], typing.Tuple[np.ndarray, np.ndarray]]:
		if self.__splitter is None:
			Logger.warning(f"Splitting disabled only processing train data.")
			return (X, y), (None, None)
		Logger.info(f"Splitting data...")
		return self.__splitter.split(X, y)

	def __process_df_batch(self, df: pd.DataFrame):
		X, y = self.__prepare_data(df)
		(train_X, train_y), (test_X, test_y) = self.__split(X, y)
		for X, y, purpose_dir in [(train_X, train_y, self.__train_dir), (test_X, test_y, self.__test_dir)]:
			if X is None:
				continue
			self.__batch_and_save(X, y, purpose_dir)

	def _batch_df(self, df: pd.DataFrame) -> typing.List[pd.DataFrame]:
		if self.__process_batch_size is None:
			return [df]
		return [
			df.iloc[i*self.__process_batch_size: (i+1)*self.__process_batch_size]
			for i in range(0, df.shape[0]//self.__process_batch_size)
		]

	def start(self, start_date: datetime = None, end_date: datetime = None):
		df = self.__filter_df(start_date, end_date)

		dfs = self._batch_df(df)
		for i, df in enumerate(dfs):
			self.__process_df_batch(df)
			Logger.info(f"Processed {i}/{len(dfs)} batches ...")

		Logger.success(f"Preparation Done!")