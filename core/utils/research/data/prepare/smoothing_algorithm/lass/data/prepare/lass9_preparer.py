import typing

import numpy as np
import pandas as pd

from core.utils.research.data.prepare.utils.data_prep_utils import DataPrepUtils
from lib.utils.logger import Logger
from .lass8_preparer import Lass8Preparer


class Lass9Preparer(Lass8Preparer):

	def __init__(
			self,
			x_columns: typing.Tuple[str, ...],
			y_columns: typing.Tuple[str, ...],
			*args,
			**kwargs
	):
		super().__init__(*args, **kwargs)
		self._x_columns = x_columns
		self.__y_columns = y_columns

	def __extract_columns(self, df: pd.DataFrame) -> np.ndarray:
		return df[list(self._x_columns)].to_numpy().transpose((1, 0))

	def _extract_columns(self, df: pd.DataFrame) -> np.ndarray:
		return self.__extract_columns(df)

	def _extract_granularity(self, data: np.ndarray, g: int) -> np.ndarray:
		df = pd.DataFrame(columns=self._x_columns, data=data.transpose((1, 0)))
		return self.__extract_columns(DataPrepUtils.condense_granularity(df, g))

	def _prepare_sequence(self, sequence: np.ndarray) -> np.ndarray:
		decomposed = np.zeros_like(sequence)
		for col in self.__y_columns:
			idx = self._x_columns.index(col)
			Logger.info(f"Decomposing {col}...")
			decomposed[idx] = self._decomposer.decompose(sequence[idx])

		return np.stack([
			sequence,
			decomposed
		], axis=0)

	def _prepare_y(self, sequences: np.ndarray) -> np.ndarray:
		y = super()._prepare_y(sequences)
		mask = [col in self.__y_columns for col in self._x_columns]
		return y[:, mask]