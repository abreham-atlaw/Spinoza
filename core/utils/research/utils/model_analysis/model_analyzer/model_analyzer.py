import random
import typing
from abc import abstractmethod, ABC

import torch
from torch import nn
from torch.utils.data import DataLoader

from core.utils.research.data.load import BaseDataset
from lib.utils.logger import Logger


class ModelAnalyzer(ABC):

	def __init__(
			self,
			model: nn.Module,
			dataset: BaseDataset,
			sample_size: int = None,
			random_state: int = 42
	):
		self._model = model
		self.__dataset = dataset
		self.__sample_size = sample_size
		self.__random_state = random_state

	@abstractmethod
	def _analyze(self, model: nn.Module, X: torch.Tensor, y: torch.Tensor, w: torch.Tensor):
		pass

	def __extract_samples(self, idxs: typing.List[int]) -> typing.Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
		Logger.info(f"Extracting {len(idxs)} Samples...")
		X, y, w = None, None, torch.zeros((len(idxs),),dtype=torch.float64)
		for i, idx in enumerate(idxs):
			Xb, yb, wb = self.__dataset[idx]
			if X is None:
				X, y = [
					torch.zeros((len(idxs), *arr.shape))
					for arr in [Xb, yb]
				]
			X[i], y[i], w[i] = Xb, yb, wb
		return X, y, w

	def __prepare_data(self):
		Logger.info(f"Preparing Data...")
		idxs = torch.arange(len(self.__dataset)).tolist()
		if self.__sample_size is not None:
			r = random.Random(self.__random_state)
			r.shuffle(idxs)
			idxs = idxs[:self.__sample_size]
		X, y, w = self.__extract_samples(idxs)
		return X, y, w

	def start(self):
		X, y, w = self.__prepare_data()
		Logger.info(f"Analyzing...")
		self._model = self._model.eval()
		self._analyze(self._model, X, y, w)

