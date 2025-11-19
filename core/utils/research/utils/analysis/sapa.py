import math
import pprint
import random
import typing
from dataclasses import dataclass

import pandas as pd
import numpy as np
from matplotlib import pyplot as plt

from core.utils.research.data.prepare.smoothing_algorithm import SmoothingAlgorithm
from lib.utils.logger import Logger
from lib.utils.math import delta


@dataclass
class Action:
	start: int
	end: int
	action: int


class SmoothingAlgorithmProfitabilityAnalyzer:

	def __init__(
			self,
			df_path: str,
			view_size: int,
			samples: int = 1,
			tp_threshold: int = 5,
			action_lag_size: typing.Union[int, typing.Tuple[int, int]] = 10,
			action_lag_count: int = int(1e3),
			units: float = 70.0,
			margin_rate: float = 0.01,
			plot_size: typing.Tuple[int, int] = (20, 10),
			plot: bool = True,
			plot_show: bool = True,
			plot_cols: int = 2,
			sample_logging: bool = True,
			granularity: int = 1
	):
		self.__df_path = df_path
		self.__view_size = view_size
		self.__tp_threshold = tp_threshold
		self.__units = units
		self.__action_lag_count = action_lag_count
		self.__margin_rate = margin_rate
		self.__plot_size = plot_size
		self.__plot = plot
		self.__plot_show = plot_show
		self.__plot_cols = plot_cols
		self.__samples = samples
		self.__sample_logging = sample_logging
		self.__granularity = granularity

		if isinstance(action_lag_size, int):
			action_lag_size = (1, action_lag_size)
		self.__action_lag_size = action_lag_size

	def __load_data(self):
		x = pd.read_csv(self.__df_path)["c"].to_numpy()[::self.__granularity]
		return x

	def __select_samples(self, sequence: np.ndarray, view_size: int, stride: int) -> np.ndarray:
		Logger.info(f"Selecting {self.__samples} Samples")
		return np.stack([
			sequence[
				-(view_size + stride*i):
				None if i == 0 else -(stride*i)
			]
			for i in range(self.__samples)
		])

	def __extract_samples(self, sequence: np.ndarray, sa: SmoothingAlgorithm) -> typing.Tuple[np.ndarray, np.ndarray]:
		Logger.info(f"Selecting samples of view_size: {self.__view_size}")

		x = self.__select_samples(sequence, view_size=self.__view_size + sa.reduction, stride=self.__view_size)
		x_sa = sa.apply_on_batch(x)
		if x.shape[1] != x_sa.shape[1]:
			x = x[:, -x_sa.shape[1]:]

		return x, x_sa

	@staticmethod
	def __extract_tps(x):
		d_x = delta(x)
		tps = np.sign(d_x[:-1]) * np.sign(d_x[1:]) == -1
		return np.arange(tps.shape[0])[tps]

	def __filter_tps(self, tpi) -> np.ndarray:
		filter_mask = np.array([True] * tpi.shape[0])
		filter_mask[:-1] = tpi[1:] - tpi[:-1] > self.__tp_threshold
		new_tpi = tpi[filter_mask]
		if new_tpi.shape[0] == tpi.shape[0]:
			return new_tpi
		return self.__filter_tps(new_tpi)

	@staticmethod
	def __extract_actions(x, tpi) -> typing.List[Action]:
		action_actions = np.sign(delta(x[tpi]))
		action_starts = tpi[:-1]
		action_ends = tpi[1:]
		return [
			Action(start=action_starts[i], end=action_ends[i], action=action_actions[i])
			for i in range(len(action_actions))
		]

	def __find_optimal_actions(self, x) -> typing.List[Action]:
		tps = self.__extract_tps(x)
		if len(tps) > 0:
			tps = self.__filter_tps(tps)
		tps = np.concatenate(([0], tps, [x.shape[0] - 1]))
		return self.__extract_actions(x, tps)

	def __get_actions_profit(self, x, actions):
		def action_profit(x, a):
			return (x[a.end] - x[a.start]) * (a.action * self.__units) / self.__margin_rate

		return np.sum([action_profit(x, a) for a in actions])

	def __lag_actions(self, x, actions):
		def generate_lag(i, mx):
			return min(i + random.randint(*self.__action_lag_size), mx)

		def lag_action(x, action):
			return Action(
				start=generate_lag(action.start, action.end),
				end=generate_lag(action.end, x.shape[0] - 1),
				action=action.action,
			)

		return [lag_action(x, action) for action in actions]

	def __shake_actions(self, x, actions):
		profits = [
			self.__get_actions_profit(x, self.__lag_actions(x, actions))
			for _ in range(self.__action_lag_count)
		]
		return np.mean(profits)

	def __analyze_sample(self, x, x_sa, sa, i):
		Logger.info(f"Analyzing Sample {i}...", end="\r" if not self.__sample_logging else "\n")

		actions = self.__find_optimal_actions(x_sa)
		optimal_profit = self.__get_actions_profit(x, actions)
		shaken_profit = self.__shake_actions(x, actions)

		if self.__sample_logging:
			Logger.info(f"Optimal Actions: \n{pprint.pformat(actions)}")
			Logger.info(f"Optimal Profit: {optimal_profit}")
			Logger.info(f"Shaken Profit: {shaken_profit}")

		if self.__plot:
			plt.title(f"{sa} (Sample={i})\nOptimal Profit: {optimal_profit}\nShaken Profit: {shaken_profit}")
			plt.plot(x, label="Original")
			plt.plot(x_sa, label=str(sa))
			for a in actions:
				plt.axvline(x=a.start, color="green" if a.action == 1 else "red")
			plt.legend()

		return optimal_profit, shaken_profit

	def analyze(self, sa: SmoothingAlgorithm) -> typing.Tuple[float, float]:
		Logger.info(f"\n\nProcessing {sa}")
		sequence = self.__load_data()

		x, x_sa = self.__extract_samples(sequence, sa)

		optimal_profits, shaken_profits = [], []
		if self.__plot:
			plt.figure(figsize=self.__plot_size)
		
		for i in range(self.__samples):
			
			if self.__plot:
				plt.subplot(math.ceil(self.__samples/self.__plot_cols), self.__plot_cols, i+1)
			
			optimal_profit, shaken_profit = self.__analyze_sample(x[i], x_sa[i], sa, i)
			optimal_profits.append(optimal_profit)
			shaken_profits.append(shaken_profit)

		if self.__plot_show:
			plt.show()
			
		optimal_profit, shaken_profit = np.mean(optimal_profits), np.mean(shaken_profits)
		Logger.success(f"Mean Optimal Profit: {optimal_profit}")
		Logger.success(f"Mean Shaken Profit: {shaken_profit}")
		Logger.success(f"Max Optimal Profit: {np.max(optimal_profits)}")
		Logger.success(f"Max Shaken Profit: {np.max(shaken_profits)}")
		Logger.success(f"Min Optimal Profit: {np.min(optimal_profits)}")
		Logger.success(f"Min Shaken Profit: {np.min(shaken_profits)}")

		return optimal_profit, shaken_profit
