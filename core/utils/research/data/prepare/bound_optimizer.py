import os
import random
import typing
import uuid

import numpy as np
import pandas as pd
from matplotlib import pyplot as plt

from core.utils.research.data.prepare import DataPreparer, SimulationSimulator, SimulationSimulator2
from core.utils.research.data.prepare.smoothing_algorithm import MovingAverage
from lib.utils.decorators import retry
from lib.utils.logger import Logger


class BoundGenerator:

	def __init__(
			self,
			start: float,
			end: float,
			csv_path: str,
			threshold=None,
			average_window=10,
			granularity=5,
			tmp_path="/tmp",
			smoothing_algorithm=None,
			prep_block_size: int = 10,
			dataprep_class: typing.Type = SimulationSimulator
	):
		self.__start = start
		self.__end = end
		self.__threshold = threshold
		self.__df = pd.read_csv(csv_path)
		self.__tmp_path = tmp_path
		self.__prep_block_size = prep_block_size

		if smoothing_algorithm is None:
			smoothing_algorithm = MovingAverage(average_window)
		self.__smoothing_algorithm = smoothing_algorithm
		self.__granularity = granularity
		self.__dataprep_class = dataprep_class

	def __prepare_tmp_path(self):
		path = os.path.join(self.__tmp_path, f"{uuid.uuid4()}.bo")
		return path

	@retry(OverflowError, patience=10)
	def __poly_generate(self, n):
		p = 2*np.random.randint(0, 100) + 1
		q = np.random.randint(n, n*10)
		f = lambda x: (((self.__end - 1) / ((q / 2) ** p)) * ((x - (q / 2)) ** p)) + 1
		return random.choices(np.array([f(x) for x in range(q)]), k=n)

	def __random_generate(self, n):
		return np.random.uniform(self.__start, self.__end, n)

	def __linear_generate(self, n):
		bound = [x + (s * ((self.__end - self.__start) / 2)) for s, x in zip([-1, 1], [self.__start, self.__end])]
		return np.linspace(*bound, n)

	def __generator(self, n):
		return random.choice([self.__poly_generate, self.__random_generate, self.__linear_generate])(n)

	def get_frequencies(self, bounds):
		path = self.__prepare_tmp_path()

		data_preparer = self.__dataprep_class(
			df=self.__df,
			bounds=bounds,
			seq_len=self.__prep_block_size,
			extra_len=1,
			batch_size=int(1e9),
			output_path=path,
			granularity=self.__granularity,
			smoothing_algorithm=self.__smoothing_algorithm,
			order_gran=False
		)

		data_preparer.start()

		y = np.concatenate([
			np.load(os.path.join(path, "train/y", f))
			for f in sorted(os.listdir(os.path.join(path, "train/y")))
		])[:, :-1]

		y_classes = np.argmax(y, axis=1)
		classes, frequencies = np.unique(y_classes, return_counts=True)

		frequencies = frequencies / np.sum(frequencies)
		os.system(f"rm -fr \"{path}\"")
		return classes, frequencies

	def __plot(self, bounds, indexes, frequencies):
		plt.figure()
		plt.scatter(indexes, frequencies)

		plt.figure()
		plt.scatter(list(range(len(bounds))), bounds)

		plt.show()

	def __filter_valid(self, bounds, threshold=None, plot=False):
		indexes, frequencies = self.get_frequencies(bounds)

		valid_bounds = [
			bounds[idx]
			for i, idx in enumerate(indexes)
			if idx < len(bounds) and (threshold is None or frequencies[i] > threshold)
		]

		if plot:
			self.__plot(valid_bounds, indexes, frequencies)

		return valid_bounds

	def __generate(self, n, bounds):
		bounds = sorted(bounds + list(self.__generator(n - len(bounds))))
		bounds = self.__filter_valid(bounds, threshold=self.__threshold)
		Logger.success(f"Found bounds: {len(bounds)}")
		if len(bounds) < n:
			bounds = self.__generate(n, bounds)
		return bounds

	def plot_bounds(self, bounds):
		self.__filter_valid(bounds, threshold=0, plot=True)

	def generate(self, n, plot=False):
		Logger.info(f"Generating {n} bounds...")
		bounds = self.__generate(n, [])

		if plot:
			self.plot_bounds(bounds)

		return bounds

	def get_weights(self, bounds: typing.List[float]) -> typing.List[float]:
		indexes, frequencies = self.get_frequencies(bounds)

		weights = np.ones((len(bounds) + 1,))
		weights[indexes] = np.sum(frequencies) / frequencies

		return list(weights)
