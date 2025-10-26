import numpy as np
import torch
import typing
from torch.utils.data import Dataset

import random
from collections import OrderedDict
import os

from core.utils.research.data.prepare.swg.abstract_swg import AbstractSampleWeightGenerator
from lib.utils.logger import Logger


class BaseDataset(Dataset):

	__NUMPY_TORCH_TYPE_MAP = {
		np.int8: torch.int8,
		np.int32: torch.int32,
		np.int64: torch.int64,
		np.uint8: torch.uint8,
		np.float16: torch.float16,
		np.float32: torch.float32,
		np.float64: torch.float64,
	}

	def __init__(
			self,
			root_dirs: list,
			cache_size: int = 5,
			X_dir: str = "X",
			y_dir: str = "y",
			w_dir: str = "w",
			out_dtypes: typing.Type = np.float64,
			num_files: typing.Optional[int] = None,
			device=torch.device("cpu"),
			check_last_file: bool = False,
			check_file_sizes: bool = False,
			load_weights: bool = False,
			return_weights: bool = True,
			swg: AbstractSampleWeightGenerator = None
	):
		self.__device = device
		self.__dtype = out_dtypes
		self.root_dirs = root_dirs
		self.__X_dir = X_dir
		self.__y_dir = y_dir
		self.__w_dir = w_dir
		self.__swg = swg
		self.random_state = None

		self.__files, self.__root_dir_map = self.__get_files(size=num_files)

		self.cache = OrderedDict()
		self.cache_size = cache_size
		self.data_points_per_file = self.__get_dp_per_file()

		if check_last_file:
			self.__check_last_file()

		if check_file_sizes:
			self.__check_file_sizes()

		self.__load_weights = load_weights
		self.__return_weights = return_weights

	@property
	def random(self):
		if self.random_state is None:
			return None
		return np.random.default_rng(self.random_state)

	@property
	def __use_swg(self) -> bool:
		return self.__swg is not None

	def set_device(self, device: torch.device):
		self.__device = device

	def shuffle(self):
		print("[+]Shuffling dataset...")
		random.shuffle(self.__files)
		self.cache = OrderedDict()
		self.random_state = random.randint(0, 1000)

	def __get_dp_per_file(self) -> int:
		first_file_name = self.__files[0]
		first_file_data = self.__load_array(os.path.join(self.root_dirs[0], self.__X_dir, first_file_name))
		return first_file_data.shape[0]

	def __check_path_file_sizes(self, path):
		Logger.info(f"Checking {path}...")
		filenames = os.listdir(os.path.join(path, self.__X_dir))

		for filename in filenames:
			file_size = self.__load_array(os.path.join(path, self.__X_dir, filename)).shape[0]
			if file_size == self.data_points_per_file:
				continue
			Logger.warning(f"[+]File({filename}) has {file_size} data points. Expected {self.data_points_per_file}, Removing...")
			self.__files.remove(filename)

	def __check_file_sizes(self):

		for root_dir in self.root_dirs:
			self.__check_path_file_sizes(root_dir)

	def __check_last_file(self):
		last_files = [
			sorted(os.listdir(os.path.join(root_dir, self.__X_dir)))[-1]
			for root_dir in self.root_dirs
		]

		for last_filename in last_files:
			last_file_size = self.__load_array(os.path.join(self.__root_dir_map[last_filename], self.__X_dir, last_filename)).shape[0]
			if last_file_size == self.data_points_per_file:
				continue
			Logger.warning(f"[+]Last file({last_filename}) has {last_file_size} data points. Expected {self.data_points_per_file}, Removing...")
			self.__files.remove(last_filename)

	def __get_files(self, size=None):
		files_map = {}
		files = []
		for root_dir in self.root_dirs:
			if size is not None and len(files) >= size:
				break
			X_encoder_path = os.path.join(root_dir, self.__X_dir)

			dir_files = sorted(os.listdir(X_encoder_path))
			if size is not None:
				dir_files = dir_files[:size - len(files)]
			files += dir_files
			for file in dir_files:
				files_map[file] = root_dir
		return files, files_map

	def __len__(self):
		return len(self.__files) * self.data_points_per_file

	def __load_array(self, path: str) -> torch.Tensor:
		out: np.ndarray = np.load(path).astype(self.__dtype)
		indexes = np.arange(out.shape[0])
		if self.random is not None:
			self.random.shuffle(indexes)

		# torch.from_numpy(dp[data_idx]).type(self.__NUMPY_TORCH_TYPE_MAP[dp.dtype])
		return torch.from_numpy(
			out[indexes]
		).to(self.__device).type(
			self.__NUMPY_TORCH_TYPE_MAP[self.__dtype]
		)

	def __load_w(self, root_dir, filename) -> torch.Tensor:
		if not self.__load_weights:
			return torch.Tensor([torch.nan for _ in range(self.data_points_per_file)])

		return self.__load_array(os.path.join(root_dir, self.__w_dir, filename))

	def __generate_weight(self, X: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
		return torch.from_numpy(self.__swg(X.numpy(), y.numpy()))

	def __load_files(self, idx: int) -> torch.Tensor:
		if idx in self.cache:
			return self.cache[idx]

		if len(self.cache) >= self.cache_size:
			self.cache.popitem(last=False)

		file_name = self.__files[idx]
		root_dir = self.__root_dir_map[file_name]

		X = self.__load_array(os.path.join(root_dir, self.__X_dir, file_name))
		y = self.__load_array(os.path.join(root_dir, self.__y_dir, file_name))
		if self.__use_swg:
			w = self.__generate_weight(X, y)
		else:
			w = self.__load_w(root_dir, file_name)

		self.cache[idx] = (X, y, w)

		return X, y, w

	def __getitem__(self, idx):
		file_idx = idx // self.data_points_per_file
		data_idx = idx % self.data_points_per_file

		X, y, w = self.__load_files(file_idx)

		value = tuple([dp[data_idx] for dp in [X, y, w]])
		if not self.__return_weights:
			value = value[:-1]
		return value
