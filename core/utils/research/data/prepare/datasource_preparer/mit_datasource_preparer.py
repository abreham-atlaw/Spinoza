import os.path
import typing
from datetime import datetime

import pandas as pd

from .datasource_preparer import DatasourcePreparer


class MITDatasourcePreparer(DatasourcePreparer):

	def __init__(self, *args, output_filename: str = "All-All.csv", **kwargs):
		super().__init__(*args, **kwargs)
		self.__output_filename = output_filename

	@staticmethod
	def __filter_intersection_time(dfs: typing.List[pd.DataFrame]):
		common_time = dfs[0]["time"]
		for df in dfs[1:]:
			common_time = common_time[common_time.isin(df["time"])]

		dfs = [df[df["time"].isin(common_time)] for df in dfs]
		return dfs

	def prepare_multiple(
			self,
			paths: typing.List[str],
			time_range: typing.Tuple[datetime, datetime]
	):
		dfs = [self._load_df(path) for path in paths]
		dfs = [self._filter_time(df, time_range) for df in dfs]
		dfs = self.__filter_intersection_time(dfs)
		dfs = [self._correct_time(df) for df in dfs]
		df = pd.concat(dfs, axis=0)
		self._export_df(df, self.__output_filename)
