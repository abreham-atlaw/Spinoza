import typing

import matplotlib.pyplot as plt
import numpy as np

from lib.utils.logger import Logger
from .rs_filter import RSFilter
from .rsa import RSAnalyzer
from ...data.collect.runner_stats import RunnerStats
from ...data.collect.runner_stats2 import RunnerStats2
from ...data.collect.runner_stats_populater import RunnerStatsPopulater


class PlotRSAnalyzer(RSAnalyzer):

	__LOSS_NAMES = [
		str(loss)
		for loss in RunnerStatsPopulater.get_evaluation_loss_functions()
	]

	def __init__(
			self,
			branches: typing.List[str],
			color_value_loss: int = 3,
			sort_loss: int = 1,
			use_avg_profits: bool = False,
			export_path: str = "plotted.csv",
			extra_filter: typing.Optional[RSFilter] = None,
			color_value_function: typing.Optional[typing.Callable] = None,
			sessions_len: int = None,
			**kwargs
	):
		rs_filter = RSFilter(
			evaluation_complete=True
		)
		if extra_filter is not None:
			rs_filter += extra_filter
		rs_filter.min_sessions = rs_filter.min_sessions or sessions_len or 1

		super().__init__(
			branches=branches,
			rs_filter=rs_filter,
			export_path=export_path,
			sort_key=lambda stat: stat.model_losses[sort_loss],
			**kwargs
		)
		self.__color_value_loss = color_value_loss
		self.__color_value_function = color_value_function
		self.__use_avg_profits = use_avg_profits
		self.__sessions_len = sessions_len

	def __trim_stat(self, stat: RunnerStats2, sessions_len: int):
		stat.sessions = stat.sessions[:sessions_len]
		return stat

	def __get_common_sessions(self, sessions: typing.List[typing.List[str]]) -> typing.List[str]:

		if None in sessions or len(sessions) == 1:
			return sessions[0]

		if len(sessions) == 2:
			return list(filter(
				lambda session: session in sessions[0] and session in sessions[1],
				sessions[0] + sessions[1]
			))

		return self.__get_common_sessions([sessions[0], self.__get_common_sessions(sessions[1:])])

	def __sync_sessions(self, stats: typing.List[RunnerStats2]):
		Logger.info(f"Syncing Sessions...")

		sessions = sorted(self.__get_common_sessions(list(map(lambda stat: stat.simulated_timestamps, stats))))

		for stat in stats:
			stat.sessions = sorted(
				list(filter(
					lambda session: session.simulated_timestamp in sessions,
					stat.sessions
				)),
				key=lambda session: sessions.index(session.simulated_timestamp)
			)

	def __trim_sessions(self, stats: typing.List[RunnerStats]):

		self.__sync_sessions(stats)

		sessions_len = self.__sessions_len
		if sessions_len is None:
			sessions_len = min(map(
				lambda stat: len(stat.session_timestamps),
				stats
			))
		Logger.info(f"Using Sessions Len = {sessions_len}")

		stats = list(map(
			lambda stat: self.__trim_stat(stat, sessions_len),
			stats
		))
		return stats

	def _get_color_values(self, stats: typing.List[RunnerStats]) -> np.ndarray:

		fn = lambda stat: stat.model_losses[self.__color_value_loss]
		if self.__color_value_function is not None:
			fn = self.__color_value_function

		colors = np.array([
			float(fn(stat))
			for stat in stats
		]).astype(np.float32)
		colors /= np.max(colors)
		return colors

	def __get_profit(self, dp: RunnerStats) -> float:
		profit = dp.profit
		if self.__use_avg_profits:
			profit = profit / len(dp.session_timestamps)
		return profit

	def _handle_stats(self, stats: typing.List[RunnerStats]):
		stats = self.__trim_sessions(stats)

		losses_count = min(map(
			lambda stat: len(stat.model_losses),
			stats
		))
		Logger.info(f"Using Losses Count = {losses_count}")

		losses = np.array([
			[stats[i].model_losses[j] for j in range(losses_count)]
			for i in range(len(stats))
		])  # (len(stats), len(losses))

		profits = np.array(list(map(
			lambda stat: self.__get_profit(stat),
			stats
		)))

		colors = self._get_color_values(stats)

		for i in range(losses_count):
			plt.figure()
			plt.title(self.__LOSS_NAMES[i])
			plt.grid(True)
			plt.scatter(
				losses[:, i],
				profits,
				c=colors,
				cmap="plasma"
			)
			plt.colorbar()
			plt.axhline(y=0, color="black")
			plt.legend()

		super()._handle_stats(stats)
		plt.show()
