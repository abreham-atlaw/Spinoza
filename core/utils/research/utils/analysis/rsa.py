import typing
from abc import ABC, abstractmethod

import numpy as np
import pandas as pd

from core.di import ResearchProvider
from lib.utils.logger import Logger
from .rs_filter import RSFilter
from ...data.collect.runner_stats import RunnerStats
from ...data.collect.runner_stats2 import RunnerStats2, RunnerStats2Session


class RSAnalyzer(ABC):

	def __init__(
			self,
			branches: typing.List[str],
			rs_filter: RSFilter,
			export_path: str,
			sort_key: typing.Callable = None,
			return_thresholds: typing.List[float] = None
	):
		self.__branches = branches
		self.__repositories = {
			branch: ResearchProvider.provide_runner_stats_repository(branch)
			for branch in branches
		}
		self.__rs_filter = rs_filter
		self.__export_path = export_path

		if sort_key is None:
			sort_key = lambda stat: stat.id
		self.__sort_key = sort_key

		self.__return_thresholds = return_thresholds if return_thresholds is not None else []

	@staticmethod
	def __filter_stats(
			stats: typing.List[RunnerStats],
			rs_filter: RSFilter
	) -> typing.List[RunnerStats]:

		if rs_filter.evaluation_complete:
			stats = [
				dp
				for dp in stats
				if 0.0 not in dp.model_losses
			]

		if rs_filter.model_key is not None:
			stats = [dp for dp in stats if rs_filter.model_key in dp.model_name]

		if rs_filter.min_sessions is not None:
			stats = [dp for dp in stats if len(dp.session_timestamps) >= rs_filter.min_sessions]

		if rs_filter.max_model_losses is not None:
			for i, max_loss in enumerate(rs_filter.max_model_losses):
				if max_loss is not None:
					stats = [dp for dp in stats if dp.model_losses[i] < max_loss]

		if rs_filter.min_model_losses is not None:
			for i, min_loss in enumerate(rs_filter.min_model_losses):
				if min_loss is not None:
					stats = [dp for dp in stats if dp.model_losses[i] > min_loss]

		if rs_filter.max_temperature is not None:
			stats = [dp for dp in stats if dp.temperature <= rs_filter.max_temperature]

		if rs_filter.min_temperature is not None:
			stats = [dp for dp in stats if dp.temperature >= rs_filter.min_temperature]

		if rs_filter.min_profit is not None:
			stats = [dp for dp in stats if dp.profit >= rs_filter.min_profit]

		if rs_filter.max_profit is not None:
			stats = [dp for dp in stats if dp.profit <= rs_filter.max_profit]

		if rs_filter.filter_fn is not None:
			stats = list(filter(rs_filter.filter_fn, stats))

		return stats


	@staticmethod
	def __attach_sessions_size(stats: typing.List[RunnerStats2]):
		for stat in stats:
			stat.sessions_size = len(stat.sessions)
		return stats

	def __get_threshold_return(self, stat: RunnerStats2, threshold: float) -> np.ndarray:
		max_profits = np.array([
			max(session.timestep_pls)
			for session in stat.sessions
		])
		total_profits = np.array(stat.profits)

		miss_mask = max_profits < threshold

		pls = total_profits
		pls[(~miss_mask)] = (threshold-1) * 100
		return pls

	def __construct_df(self, stats: typing.List[RunnerStats2]) -> pd.DataFrame:
		return pd.DataFrame([
			(
				stat.id, stat.temperature, stat.profit, stat.model_losses,
				[dt.strftime("%Y-%m-%d %H:%M:%S.%f") for dt in stat.session_timestamps],
				stat.profits, stat.simulated_timestamps, stat.session_model_losses,
				stat.sessions_size,
				[
					max(session.timestep_pls)
					for session in stat.sessions
				],
				[
					min(session.timestep_pls)
					for session in stat.sessions
				],
				[
					session.timestep_pls
					for session in stat.sessions
				],
			) + tuple([
				list(self.__get_threshold_return(stat, threshold))
				for threshold in self.__return_thresholds
			]) + tuple([
				sum(self.__get_threshold_return(stat, threshold))
				for threshold in self.__return_thresholds
			])
			for stat in stats
		], columns=[
			"ID", "Temperature", "Profit", "Losses", "Sessions", "Profits", "Sim. Timestamps",
			"Session Model Losses", "Sessions Size", "Max Profit", "Min Profit", "TimeStep Profits"
		] + [
			f"Profits(Return Threshold: {threshold})"
			for threshold in self.__return_thresholds
		] + [
			f"Profit(Return Threshold: {threshold})"
			for threshold in self.__return_thresholds
		]
		)

	def _export_stats(self, stats: typing.List[RunnerStats], path: str):
		Logger.info(f"Exporting {len(stats)} stats to {path}")
		df = self.__construct_df(stats)
		df.to_csv(path)

	def _handle_stats(self, stats: typing.List[RunnerStats]):
		self._export_stats(stats, self.__export_path)

	def _process(self):

		Logger.info(f"Collecting stats for {self.__branches}")
		stats = [
			stat
			for repository in self.__repositories.values()
			for stat in repository.retrieve_all()
		]

		self.__attach_sessions_size(stats)

		Logger.info(f"Filtering {len(stats)} stats")
		stats = self.__filter_stats(
			stats,
			self.__rs_filter
		)

		Logger.info(f"Sorting {len(stats)} stats")
		stats = sorted(
			stats,
			key=self.__sort_key
		)

		Logger.info(f"Handling {len(stats)} stats")
		self._handle_stats(stats)

		Logger.info(f"Done!")

	def start(self):
		self._process()

