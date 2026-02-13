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
			session_take_profits: typing.List[float] = None,
			session_stop_losses: typing.List[float] = None
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

		self.__session_take_profits = session_take_profits if session_take_profits is not None else []
		self.__session_stop_losses = session_stop_losses if session_stop_losses is not None else []

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

	@staticmethod
	def __calc_triggered_profit(session: RunnerStats2Session, sl: float, tp: float) -> float:
		def filter_take_profit(profits, tp):
			mask = np.cumprod(np.concatenate([[True], profits < tp])[:-1]).astype(bool)
			return profits[mask]

		def filter_stop_loss(profits, sl):
			mask = np.cumprod(np.concatenate([[True], profits > sl])[:-1]).astype(bool)
			return profits[mask]

		profits = filter_stop_loss(
			filter_take_profit(np.array(session.timestep_pls), tp),
			sl
		)

		return (profits[-1] - 1)*100

	def __get_triggered_returns(self, stat: RunnerStats2, sl: float, tp: float) -> typing.List[float]:
		return [
			self.__calc_triggered_profit(session, sl, tp)
			for session in stat.sessions
		]

	def __construct_df(self, stats: typing.List[RunnerStats2]) -> pd.DataFrame:
		return pd.DataFrame([
			(
				stat.id, stat.temperature, stat.profit, stat.model_losses,
				[dt.strftime("%Y-%m-%d %H:%M:%S.%f") for dt in stat.session_timestamps],
				stat.profits, stat.simulated_timestamps, stat.session_model_losses,
				stat.sessions_size,
				[
					max(session.timestep_pls) if len(session.timestep_pls) > 0 else 0
					for session in stat.sessions
				],
				[
					min(session.timestep_pls) if len(session.timestep_pls) > 0 else 0
					for session in stat.sessions
				],
				[
					session.timestep_pls
					for session in stat.sessions
				],
			) + tuple([
				self.__get_triggered_returns(stat, sl, tp)
				for tp in self.__session_take_profits
				for sl in self.__session_stop_losses
			]) + tuple([
				sum(self.__get_triggered_returns(stat, sl, tp))
				for tp in self.__session_take_profits
				for sl in self.__session_stop_losses
			])
			for stat in stats
		], columns=[
			"ID", "Temperature", "Profit", "Losses", "Sessions", "Profits", "Sim. Timestamps",
			"Session Model Losses", "Sessions Size", "Max Profit", "Min Profit", "TimeStep Profits"
		] + [
			f"Profits(Take Profit: {tp}, Stop Loss: {sl})"
			for tp in self.__session_take_profits
			for sl in self.__session_stop_losses
		] + [
			f"Profit(Take Profit: {tp}, Stop Loss: {sl})"
			for tp in self.__session_take_profits
			for sl in self.__session_stop_losses
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

