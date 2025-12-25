import os.path
import typing

import numpy as np
import pandas as pd
import torch
from matplotlib import pyplot as plt
from torch import nn
from torch.utils.data import DataLoader

from core import Config
from core.environment.trade_state import TradeState
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
			bounds: typing.Iterable[float] = None,
			extra_len: int = 124
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
		self.__extra_len = extra_len

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

	def plot_sequence(self, instrument: typing.Tuple[str, str], checkpoints: typing.List[int] = None, new_figure=True):
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

		if new_figure:
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
		if new_figure:
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
		y_hat = self.__softmax(model(torch.from_numpy(X))[..., :-1]).detach().numpy()
		return y_hat

	def __get_yv(self, y: np.ndarray) -> np.ndarray:
		bounds = DataPrepUtils.apply_bound_epsilon(self.__bounds)
		bounds = (bounds[1:] + bounds[:-1])/2
		return np.sum(y[:, :-1] * bounds, axis=1)

	@staticmethod
	def __evaluate_samples_loss(y_hat: np.ndarray, y: np.ndarray, loss_fn: SpinozaLoss) -> np.ndarray:
		if loss_fn.collapsed:
			Logger.error("Calling SessionAnalyzer.__evaluate_samples_loss with collapsed loss function")

		loss = loss_fn(*[
			torch.from_numpy(arr)
			for arr in [y_hat, y]
		])
		return loss.detach().numpy()

	def plot_timestep_output(
			self,
			i: int,
			h: float = 0.0,
			max_depth: int = 0,
			loss: SpinozaLoss = None,
			instrument: typing.Tuple[str, str] = None
	):

		if instrument is None:
			instrument = self.__instruments[0]

		X, y = self.__load_output_data()
		y_hat = self.__get_y_hat(X, h=h, max_depth=max_depth)

		l = self.__evaluate_samples_loss(y_hat=y_hat, y=y, loss_fn=loss) if loss is not None else None

		y_v, y_hat_v = [self.__get_yv(_y) for _y in [y, y_hat]]

		plt.figure(figsize=self.__fig_size)

		plt.subplot(1, 2, 1)
		plt.title(f"Timestep Output - i={i}, h={h}, max_depth={max_depth}")
		plt.grid()

		if X.ndim == 2:
			X = np.expand_dims(X, axis=1)

		X = X[..., :X.shape[-1] - self.__extra_len]
		for c in range(X.shape[1]):
			plt.plot(X[i, c][X[i, c] > 0], label=f"Channel: {c}")
		plt.legend()

		plt.subplot(1, 2, 2)
		plt.title(f"""y: {y_v[i]}
y_hat: {y_hat_v[i]}
loss: {l[i] if l is not None else "N/A"}
""")
		plt.plot(y[i, :-1], label="Y")
		plt.plot(y_hat[i, :-1], label="Y-hat")
		plt.legend()
		plt.show()

	def __prepare_node_input_data(self, seq: np.ndarray) -> torch.Tensor:
		extra_len = self.__model.input_size[-1] - seq.shape[-1]
		dtype = next(self.__model.parameters()).dtype

		X = np.expand_dims(
			np.concatenate(
				(seq, np.zeros((seq.shape[0], extra_len))),
				axis=-1
			),
			axis=0
		)

		if len(self.__model.input_size) == 2:
			X = np.squeeze(X, axis=1)

		return torch.from_numpy(X).type(dtype)

	def plot_node_prediction(
			self, 
			idx: int,
			path: typing.List[int] = None,
			instrument: typing.Tuple[str, str] = None,
			channels: typing.List[int] = None
	):
		if path is None:
			path = []

		if instrument is None:
			instrument = self.__instruments[0]

		node, repo = self.load_node(idx)
		node = self.get_node(node, path)

		state: TradeState = repo.retrieve(node.id)
		seq = state.get_market_state().get_channels_state(*instrument)

		if channels is None:
			channels = np.arange(seq.shape[0])

		X = self.__prepare_node_input_data(seq)
		y_hat = self.__softmax(self.__model(X)[..., :-1]).detach().numpy()

		if y_hat.ndim == 2:
			y_hat = np.expand_dims(y_hat, axis=1)

		plt.figure(figsize=self.__fig_size)
		plt.subplot(1, 2, 1)
		plt.grid()
		plt.title(f"Node Prediction - idx={idx}, path={path}, depth={len(path)//2}")

		for i in channels:
			plt.plot(seq[i][seq[i] > 0], label=f"Channel: {i}")

		if len(path) > 0:
			plt.axvline(x=seq.shape[1] - (len(path)//2) - 1, color="red")

		plt.legend()

		plt.subplot(1, 2, 2)
		plt.title(f"y_hat={[self.__get_yv(y_hat[:, c])[0] for c in range(y_hat.shape[1])]}")
		for c in range(y_hat.shape[1]):
			plt.plot(y_hat[0, c], label=f"Channel: {c}")

		plt.show()

	@CacheDecorators.cached_method()
	def __load_prediction_sequence_input_data(self, instrument: typing.Tuple[str, str]) -> np.ndarray:

		X = None

		for i in range(len(os.listdir(self.__graphs_path))):
			node, repo = self.load_node(i)
			state: TradeState = repo.retrieve(node.id)
			x = state.get_market_state().get_channels_state(*instrument)
			x = self.__prepare_node_input_data(x).numpy()
			if X is None:
				X = x
				continue
			X = np.concatenate((X, x), axis=0)

		return X

	def plot_prediction_sequence(
			self,
			instrument: typing.Tuple[str, str] = None,
			channel: int = 0,
			h: float = 0.0,
			max_depth: int = 0,
	):
		if instrument is None:
			instrument = self.__instruments[0]

		X = self.__load_prediction_sequence_input_data(instrument)
		y_hat = self.__get_y_hat(X, h=h, max_depth=max_depth)

		y_hat = np.expand_dims(y_hat, axis=1) if y_hat.ndim == 2 else y_hat

		y_hat = y_hat[:, channel]

		y_hat_v = self.__get_yv(y_hat) - 1

		labels = [str(i) for i in range(y_hat_v.shape[0])]
		colors = ["green" if v >= 0 else "red" for v in y_hat_v]
		max_val = max(abs(v) for v in y_hat_v)

		plt.figure(figsize=self.__fig_size)

		plt.subplot(2, 1, 1)
		self.plot_sequence(instrument, new_figure=False)

		plt.subplot(2, 1, 2)
		plt.title(f"Prediction Sequence of {instrument} on Channel {channel}")
		plt.bar(labels, y_hat_v, color=colors)
		plt.xticks(rotation=90, fontsize=5)

		plt.axhline(y=0, color="black")
		plt.ylim(-max_val, max_val)


		plt.xlabel("Timestep")
		plt.ylabel("Prediction")

		plt.show()
