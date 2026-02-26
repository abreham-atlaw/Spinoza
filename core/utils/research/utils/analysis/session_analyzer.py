import os.path
import sqlite3
import typing
from dataclasses import dataclass
from datetime import datetime

import numpy as np
import pandas as pd
import torch
from matplotlib import pyplot as plt
from torch import nn

from core import Config
from core.di import AgentUtilsProvider
from core.environment.trade_state import TradeState
from core.utils.research.data.prepare.smoothing_algorithm import SmoothingAlgorithm
from core.utils.research.data.prepare.utils.data_prep_utils import DataPrepUtils
from core.utils.research.losses import SpinozaLoss
from core.utils.research.model.model.savable import SpinozaModule
from core.utils.research.model.model.utils import HorizonModel, AggregateModel
from core.utils.research.utils.model_evaluator import ModelEvaluator
from lib.rl.agent import Node
from lib.rl.agent.utils.state_predictor import StatePredictor
from lib.utils.cache import Cache
from lib.utils.cache.decorators import CacheDecorators
from lib.utils.logger import Logger
from lib.utils.staterepository import StateRepository
from lib.utils.torch_utils.model_handler import ModelHandler
from temp import stats


@dataclass
class Checkpoint:
	start: int
	end: int
	take_profit: float
	stop_loss: float
	note: str
	note_color: str = "black"


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
			extra_len: int = 124,
			aggregate_alpha: float = None,
			log_bounds: bool = True,
			state_predictor: StatePredictor = None
	):
		self.__sessions_path = session_path
		self.__fig_size = fig_size
		self.set_smoothing_algorithms(smoothing_algorithms)
		self.__plt_y_grid_count = plt_y_grid_count
		self.__cache = Cache()
		self.__instruments = instruments
		if bounds is None:
			bounds = Config.AGENT_STATE_CHANGE_DELTA_STATIC_BOUND
		self.__bounds = bounds

		self.__model = model or self.__load_session_model(model_key, aggregate_alpha)
		self.__dtype = dtype
		self.__softmax = nn.Softmax(dim=-1)
		self.__extra_len = extra_len
		self.__log_bounds = log_bounds
		self.__state_predictor: StatePredictor = self.__load_state_predictor(model_key, aggregate_alpha) if state_predictor is None else state_predictor

	def __load_state_predictor(self, model_key: str, aggregate_alpha: float) -> StatePredictor:
		Config.CORE_MODEL_CONFIG.path = self.__get_model_path(model_key)
		Config.AGENT_MODEL_USE_AGGREGATION = aggregate_alpha is not None
		Config.AGENT_MODEL_AGGREGATION_ALPHA = aggregate_alpha
		return AgentUtilsProvider.provide_state_predictor()

	def __get_model_path(self, model_key: str):
		model_path = os.path.join(
			self.__sessions_path,
			next(filter(
				lambda filename: filename.endswith(".zip") and model_key in filename,
				os.listdir(self.__sessions_path)
			))
		)
		return model_path

	def __load_session_model(self, model_key: str, aggregate_alpha: float) -> SpinozaModule:
		model_path = self.__get_model_path(model_key)
		Logger.info(f"Using session model: {os.path.basename(model_path)}")
		model = ModelHandler.load(model_path)

		if aggregate_alpha is not None:
			model = AggregateModel(
				model=model,
				bounds=self.__bounds,
				a=aggregate_alpha,
				softmax=True
			)
		return model

	@property
	def __candlesticks_path(self) -> str:
		return os.path.join(self.__sessions_path, "candlesticks")

	@property
	def __graphs_path(self) -> str:
		return os.path.join(self.__sessions_path, "graph_dumps")

	@property
	def __data_path(self) -> str:
		return os.path.join(self.__sessions_path, "outs")

	@property
	def __db_path(self) -> str:
		return os.path.join(self.__sessions_path, "oanda-simulation/db.sqlite3")

	def __get_all_df_files(self) -> typing.List[str]:
		files = []
		unique_keys = []

		for i, filename in enumerate(sorted(filter(lambda fn: fn.endswith(".csv"), os.listdir(self.__candlesticks_path)))):
			filepath = os.path.join(self.__candlesticks_path, filename)
			df = pd.read_csv(filepath)
			key = df.iloc[-1]["time"], df.iloc[-1]["c"]
			if key in unique_keys:
				Logger.warning(f"Identified {filepath} as duplicate. Skipping file...")
				continue
			files.append(filepath)
			unique_keys.append(key)

		return files

	def __get_df_files(self, instrument: typing.Tuple[str, str]) -> typing.List[str]:

		idx = self.__instruments.index(instrument)
		all_files = self.__get_all_df_files()
		return all_files[idx::len(self.__instruments)]

	@CacheDecorators.cached_method()
	def __load_db(self) -> pd.DataFrame:
		cnx = sqlite3.connect(self.__db_path)
		df = pd.read_sql_query("SELECT * FROM core_trade", cnx)
		df["open_time"] = pd.to_datetime(df["open_time"])
		df["close_time"] = pd.to_datetime(df["close_time"])
		return df

	@CacheDecorators.cached_method()
	def __load_dfs(self, instrument: typing.Tuple[str, str]) -> pd.DataFrame:
		dfs = list(filter(
			lambda df: df.shape[0] > 0,
			[
				pd.read_csv(os.path.join(self.__candlesticks_path, f))
				for f in self.__get_df_files(instrument)
			]
		))
		for df in dfs:
			df["time"] = pd.to_datetime(df["time"])
		return dfs

	@CacheDecorators.cached_method()
	def __get_sequences(self, instrument: typing.Tuple[str, str], channel: str = "c") -> typing.List[np.ndarray]:
		dfs = self.__load_dfs(instrument=instrument)
		return [
			df[channel].to_numpy()
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

	def plot_sequence(
			self,
			instrument: typing.Tuple[str, str],
			checkpoints: typing.List[typing.Union[int, typing.Tuple[int, int], Checkpoint]] = None,
			new_figure=True,
			channels: typing.Tuple[str]= ('c',)
	):
		if checkpoints is None:
			checkpoints = []

		for i in range(len(checkpoints)):
			if isinstance(checkpoints[i], int):
				checkpoints[i] = (checkpoints[i], None)
			if isinstance(checkpoints[i], tuple):
				checkpoints[i] = Checkpoint(
					start=checkpoints[i][0],
					take_profit=checkpoints[i][1],
					stop_loss=None,
					end=checkpoints[i][0] + 3,
					note=f"{checkpoints[i][0]}"
				)


		x = np.array([
			[
				seq[-1]
				for seq in self.__get_sequences(instrument=instrument, channel=c)
			]
			for c in channels
		])
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


		for i, channel in enumerate(channels):
			plt.plot(x[i], label=f"Channel:{channel} - Clean")
		for i in range(len(self.__smoothing_algorithms)):
			plt.plot(x_sa[i], label=str(self.__smoothing_algorithms[i]))

		for y in np.linspace(np.min(x), np.max(x), self.__plt_y_grid_count):
			plt.axhline(y=y, color="black")

		for checkpoint in checkpoints:
			plt.axvline(x=int(checkpoint.start), color="blue")
			plt.axvline(x=int(checkpoint.start) + 1, color="green")
			plt.text(
				checkpoint.start, np.random.random() * (np.max(x) - np.min(x)) + np.min(x),
				checkpoint.note,
				verticalalignment="center",
				fontweight="bold",
				color=checkpoint.note_color,
				zorder=20
			)

			for trigger, color in zip([checkpoint.take_profit, checkpoint.stop_loss], ["green", "red"]):
				if trigger is not None:
					plt.plot([checkpoint.start+1, checkpoint.end], [trigger, trigger], zorder=10, color=color, linewidth=5)

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
		bounds = DataPrepUtils.apply_bound_epsilon(self.__bounds, log=self.__log_bounds)
		return np.sum(y[:] * bounds, axis=1)

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

	@CacheDecorators.cached_method()
	def __load_prediction_states(self) -> typing.List[TradeState]:
		states = []

		for i in range(len(os.listdir(self.__graphs_path))):
			node, repo = self.load_node(i)
			state: TradeState = repo.retrieve(node.id)
			states.append(state)

		Logger.success(f"Loaded {len(states)} states")
		return states

	def plot_prediction_sequence(
			self,
			instrument: typing.Tuple[str, str] = None,
			channels: typing.Tuple[int] = (0,),
			checkpoints: typing.List[int] = None,
	):


		if instrument is None:
			instrument = self.__instruments[0]

		if checkpoints is None:
			checkpoints = []

		states = self.__load_prediction_states()
		y_hat = self.__state_predictor.predict(
			states=states,
			actions=[None]*len(states),
			instrument=instrument
		)[..., :-1]

		y_hat = np.expand_dims(y_hat, axis=1) if y_hat.ndim == 2 else y_hat

		y_hat = y_hat[:, channels]

		y_hat_v = np.stack([
			self.__get_yv(y_hat[:, c]) - 1
			for c in channels
		], axis=1)

		labels = [str(i) for i in range(y_hat_v.shape[0])]
		colors = np.array([ ["green" if y_hat_v[i, j] >=0 else "red" for j in range(y_hat_v.shape[1])] for i in range(y_hat_v.shape[0])])
		max_val = np.max(np.abs(y_hat_v))

		plt.figure(figsize=self.__fig_size)

		plt.subplot(2, 1, 1)
		self.plot_sequence(instrument, new_figure=False, checkpoints=checkpoints)

		plt.subplot(2, 1, 2)
		plt.title(f"Prediction Sequence of {instrument} on Channels {channels}")

		x = np.arange(len(labels))
		width = 0.7 / len(channels)
		padding = 0.1
		for i in channels:
			plt.bar(x + width*i+padding, y_hat_v[:, i], width=width, color=colors[:, i], label=i)
		plt.xticks(rotation=90, fontsize=5, labels=labels, ticks=x)

		plt.axhline(y=0, color="black")
		plt.ylim(-max_val, max_val)

		plt.xlabel("Timestep")
		plt.ylabel("Prediction")

		for checkpoint in checkpoints:
			plt.axvline(x=checkpoint, color="blue")

		plt.show()

	@staticmethod
	def __get_timestep(time: datetime, dfs: typing.List[pd.DataFrame]) -> int:
		times = [df["time"].iloc[-1] for df in dfs]
		gran = round(np.mean([
			(times[i+1] - times[i]).total_seconds() // 60
			for i in range(len(times) - 1)
		]))

		for i, t in enumerate(times):
			if t.replace(tzinfo=None) > time.replace(tzinfo=None):
				return i-1 + (time.replace(tzinfo=None) - times[i-1].replace(tzinfo=None)).total_seconds() / (gran * 60)
		return len(times)

	def plot_trades(
			self,
			channels: typing.Tuple[str,...] = ("c",),
			instrument: typing.Tuple[str, str] = None
	):


		def parse_action(row):
			return "BUY" if row["units"] > 0 else "SELL"

		if instrument is None:
			instrument = self.__instruments[0]

		db = self.__load_db()
		db = db[(db["base_currency"] == instrument[0]) & (db["quote_currency"] == instrument[1])]

		dfs = self.__load_dfs(instrument=instrument)



		checkpoints = [
			Checkpoint(
				start=self.__get_timestep(row["open_time"], dfs) - 1,
				end=self.__get_timestep(row["close_time"], dfs) ,
				take_profit=row["take_profit"],
				stop_loss=row["stop_loss"],
				note=f"{int(self.__get_timestep(row['open_time'], dfs)) - 1}.\n{parse_action(row)}\n{row['realized_pl'] :.2f}",
				note_color="darkgreen" if row["realized_pl"] > 0 else "darkred"
			)
			for i, row in db.iterrows()
		]

		self.plot_sequence(
			instrument=instrument,
			channels=channels,
			checkpoints=checkpoints,
		)

