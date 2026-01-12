import typing

import numpy as np
import pandas as pd

from lib.utils.logger import Logger


class DataPrepUtils:

	@staticmethod
	def find_bound_index(bounds: typing.List[float], value: float) -> int:
		return np.sum(value >= np.array(bounds))

	@staticmethod
	def apply_bound_epsilon(bounds: typing.List[float], eps: float = None) -> typing.List[float]:
		if (isinstance(bounds, np.ndarray) and bounds.ndim == 2) or (isinstance(bounds, list) and isinstance(bounds[0], list)):
			return np.stack([
				DataPrepUtils.apply_bound_epsilon(bounds[i], eps)
				for i in range(len(bounds))
			])

		if eps is None:
			eps = (bounds[1] - bounds[0] + bounds[-1] - bounds[-2]) / 2
		Logger.info(f"Using epsilon: {eps}")
		bounds = np.concatenate([
			np.array([max(bounds[0] - eps, 1e-9)]),
			bounds,
			np.array([bounds[-1] + eps])
		])
		bounds = (bounds[1:] + bounds[:-1]) / 2
		return bounds

	@staticmethod
	def get_instruments(df: pd.DataFrame) -> typing.List[typing.Tuple[str, str]]:
		return list(df[["base_currency", "quote_currency"]].drop_duplicates().itertuples(index=False, name=None))

	@staticmethod
	def clean_df(df: pd.DataFrame) -> pd.DataFrame:
		Logger.info(f"Cleaning DataFrame")

		instruments = DataPrepUtils.get_instruments(df)
		if len(instruments) > 1:
			Logger.info(f"Found {len(instruments)} instruments: {instruments}")
			cleaned_dfs = []
			for base_currency, quote_currency in instruments:
				df_instrument = df[(df["base_currency"] == base_currency) & (df["quote_currency"] == quote_currency)].copy()
				cleaned_dfs.append(DataPrepUtils.clean_df(df_instrument))
			return pd.concat(cleaned_dfs)

		Logger.info(f"Cleaning {instruments[0]}...")
		df["time"] = pd.to_datetime(df["time"])
		df = df.drop_duplicates(subset="time")
		df = df.sort_values(by="time")
		return df

	@staticmethod
	def stack(sequence: np.ndarray, length) -> np.ndarray:
		stack = np.zeros((sequence.shape[-1] - length + 1,) + sequence.shape[:-1] + (length, ))
		for i in range(stack.shape[0]):
			stack[i] = sequence[..., i: i + length]
		return stack

	@staticmethod
	def condense_granularity(df: pd.DataFrame, g: int) -> pd.DataFrame:

		df = df.iloc[:g*(df.shape[0] // g)]
		df_g = df.iloc[0::g].copy()

		for col, condenser in zip(["l", "h", "v"], [np.min, np.max, np.sum]):
			if col not in df_g.columns:
				continue
			df_g[col] = condenser(df[col].to_numpy().reshape((-1, g)), axis=1)

		return df_g

	@staticmethod
	def encode_timestamp(df: pd.DataFrame) -> pd.DataFrame:
		timestamp = pd.to_datetime(df["time"].to_numpy())
		df[["time.year", "time.month", "time.day", "time.hour", "time.minute", "time.second"]] = np.stack(
			[timestamp.year, timestamp.month, timestamp.day, timestamp.hour, timestamp.minute, timestamp.second],
			axis=1)
		return df
