import os.path
import typing

import numpy as np
import pandas as pd
import torch
from matplotlib import pyplot as plt
from torch import nn
from torch.utils.data import DataLoader

from core import Config
from core.utils.research.data.load import BaseDataset
from core.utils.research.data.prepare.smoothing_algorithm import SmoothingAlgorithm
from core.utils.research.data.prepare.utils.data_prep_utils import DataPrepUtils
from core.utils.research.losses import SpinozaLoss
from core.utils.research.model.model.savable import SpinozaModule
from core.utils.research.model.model.utils import HorizonModel
from core.utils.research.utils.model_evaluator import ModelEvaluator
from lib.rl.agent import Node
from lib.utils.cache import Cache
from lib.utils.cache.decorators import CacheDecorators
from lib.utils.logger import Logger
from lib.utils.staterepository import StateRepository
from lib.utils.torch_utils.model_handler import ModelHandler
from temp import stats


class SessionAnalyzer:

	def __init__(
			self,
			session_path: str,
			smoothing_algorithms: typing.List[SmoothingAlgorithm],
			instruments: typing.List[typing.Tuple[str, str]],
			fig_size: typing.Tuple[int, int] = (20, 10),
			plt_y_grid_count: int = 100,
			model: typing.Optional[SpinozaModule] = None,
			dtype: typing.Type = np.float32,
			model_key: str = "spinoza-training",
			bounds: typing.Iterable[float] = None
	):
		self.__sessions_path = session_path
		self.__fig_size = fig_size
		self.set_smoothing_algorithms(smoothing_algorithms)
		self.__plt_y_grid_count = plt_y_grid_count
		self.__cache = Cache()
		self.__instruments = instruments
		self.__model = model or self.__load_session_model(model_key)
		self.__dtype = dtype

		if bounds is None:
			bounds = Config.AGENT_STATE_CHANGE_DELTA_STATIC_BOUND
		self.__bounds = bounds

		self.__softmax = nn.Softmax(dim=-1)

	def __load_session_model(self, model_key: str) -> SpinozaModule:
		model_path = os.path.join(
			self.__sessions_path,
			next(filter(
				lambda filename: filename.endswith(".zip") and model_key in filename,
				os.listdir(self.__sessions_path)
			))
		)
		Logger.info(f"Using session model: {os.path.basename(model_path)}")
		return ModelHandler.load(model_path)

	@property
	def __candlesticks_path(self) -> str:
		return os.path.join(self.__sessions_path, "candlesticks")

	@property
	def __graphs_path(self) -> str:
		return os.path.join(self.__sessions_path, "graph_dumps")

	@property
	def __data_path(self) -> str:
		return os.path.join(self.__sessions_path, "outs")

	def __get_df_files(self, instrument: typing.Tuple[str, str]) -> typing.List[str]:

		idx = self.__instruments.index(instrument)
		all_files = [
			os.path.join(self.__candlesticks_path, file)
			for file in filter(
				lambda fn: fn.endswith(".csv"),
				sorted(os.listdir(self.__candlesticks_path))
			)
		]
		return all_files[idx::len(self.__instruments)]

	@CacheDecorators.cached_method()
	def __load_dfs(self, instrument: typing.Tuple[str, str]) -> pd.DataFrame:
		return list(filter(
			lambda df: df.shape[0] > 0,
			[
				pd.read_csv(os.path.join(self.__candlesticks_path, f))
				for f in self.__get_df_files(instrument)
			]
		))

	@CacheDecorators.cached_method()
	def __get_sequences(self, instrument: typing.Tuple[str, str]) -> typing.List[np.ndarray]:
		dfs = self.__load_dfs(instrument=instrument)
		return [
			df["c"].to_numpy()
			for df in dfs
		]

	@CacheDecorators.cached_method()
	def __get_smoothed_sequences(self, instrument: typing.List[str]) -> typing.List[typing.List[np.ndarray]]:
		x = self.__get_sequences(instrument)
		return [
			[
				sa(seq)
				for seq in x
			]
			for sa in self.__smoothing_algorithms
		]

	def set_smoothing_algorithms(self, smoothing_algorithms: typing.List[SmoothingAlgorithm]):
		self.__smoothing_algorithms = smoothing_algorithms
		Logger.info("Using Smoothing Algorithms: {}".format(', '.join([str(sa) for sa in smoothing_algorithms])))

	def plot_sequence(self, instrument: typing.Tuple[str, str], checkpoints: typing.List[int] = None):
		if checkpoints is None:
			checkpoints = []

		x = [
			seq[-1]
			for seq in self.__get_sequences(instrument=instrument)
		]
		x_sa = [
			[
				smoothed_sequence[-1]
				for smoothed_sequence in sa_sequences
			]
			for sa_sequences in self.__get_smoothed_sequences(instrument=instrument)
		]

		plt.figure(figsize=self.__fig_size)
		plt.title(" / ".join(instrument))
		plt.grid()

		plt.plot(x, label="Clean")
		for i in range(len(self.__smoothing_algorithms)):
			plt.plot(x_sa[i], label=str(self.__smoothing_algorithms[i]))

		for y in np.linspace(np.min(x), np.max(x), self.__plt_y_grid_count):
			plt.axhline(y=y, color="black")

		for checkpoint in checkpoints:
			plt.axvline(x=checkpoint, color="blue")
			plt.axvline(x=checkpoint+1, color="green")
			plt.axvline(x=checkpoint+2, color="red")
			plt.text(checkpoint, max(x), str(checkpoint), verticalalignment="center")

		plt.legend()
		plt.show()

	def plot_timestep_sequence(self, instrument: typing.Tuple[str, str], i: int):
		x = self.__get_sequences(instrument=instrument)[i]
		x_sa = [
			sequence[i]
			for sequence in self.__get_smoothed_sequences(instrument=instrument)
		]

		plt.figure(figsize=self.__fig_size)
		plt.title(f"{instrument[0]} / {instrument[1]}  -  i={i}")
		plt.grid()

		plt.plot(x, label="Clean")
		for i in range(len(self.__smoothing_algorithms)):
			plt.plot(x_sa[i], label=str(self.__smoothing_algorithms[i]))

		plt.legend()
		plt.show()

	def evaluate_loss(self, loss: SpinozaLoss) -> float:
		evaluator = ModelEvaluator(
			data_path=self.__data_path,
			cls_loss_fn=loss,
		)
		return evaluator(self.__model)[0]

	def load_node(self, idx) -> typing.Tuple[Node, StateRepository]:
		return stats.load_node_repo(os.path.join(self.__graphs_path, sorted(os.listdir(self.__graphs_path))[idx]))

	@staticmethod
	def get_node(root: Node, path: typing.List[int]):
		path = path.copy()
		node = root
		while len(path) > 0:
			node = node.get_children()[path.pop(0)]
		return node

	def plot_node(self, idx: int, path: typing.List[int] = None, depth: int = None):
		node, repo = self.load_node(idx)
		if path is not None:
			node = self.get_node(node, path)
		print(f"Max Depth: {stats.get_max_depth(node)}")
		plt.figure(figsize=self.__fig_size)
		stats.draw_graph_live(node, visited=True, state_repository=repo, depth=depth)
		plt.show()

	@CacheDecorators.cached_method()
	def __load_output_data(self) -> typing.Tuple[np.ndarray, np.ndarray]:
		X, y = [
			np.concatenate([
				np.load(os.path.join(self.__data_path, axis, filename)).astype(self.__dtype)
				for filename in sorted(os.listdir(os.path.join(self.__data_path, axis)))
			]).astype(self.__dtype)
			for axis in ["X", "y"]
		]
		return X, y[:, :-1]

	@CacheDecorators.cached_method()
	def __get_y_hat(self, X: np.ndarray, h: float, max_depth: int) -> np.ndarray:
		model = self.__model
		if h > 0 and max_depth > 0:
			model = HorizonModel(
				model=self.__model,
				h=h,
				max_depth=max_depth,
				bounds=self.__bounds
			)
		y_hat = self.__softmax(model(torch.from_numpy(X))[:, :-1]).detach().numpy()
		return y_hat

	def __get_yv(self, y: np.ndarray) -> np.ndarray:
		bounds = DataPrepUtils.apply_bound_epsilon(self.__bounds)
		bounds = (bounds[1:] + bounds[:-1])/2
		return np.sum(y[:, :-1] * bounds, axis=1)

	def plot_timestep_output(
			self,
			i: int,
			h: float = 0.0,
			max_depth: int = 0,
	):
		X, y = self.__load_output_data()
		y_hat = self.__get_y_hat(X, h=h, max_depth=max_depth)

		y_v, y_hat_v = [self.__get_yv(_y) for _y in [y, y_hat]]

		plt.figure(figsize=self.__fig_size)

		plt.subplot(1, 2, 1)
		plt.title(f"Timestep Output - i={i}, h={h}, max_depth={max_depth}")
		plt.plot(X[i, :-124])

		plt.subplot(1, 2, 2)
		plt.title(f"""y: {y_v[i]}
y_hat: {y_hat_v[i]}
""")
		plt.plot(y[i, :-1], label="Y")
		plt.plot(y_hat[i, :-1], label="Y-hat")
		plt.legend()
		plt.show()
