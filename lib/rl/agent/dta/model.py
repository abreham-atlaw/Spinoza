from abc import ABC, abstractmethod

import numpy as np
import torch
import torch.nn as nn

from lib.utils.logger import Logger
from lib.utils.torch_utils.model_handler import ModelHandler


class Model(ABC):

	@abstractmethod
	def predict(self, inputs: np.ndarray) -> np.ndarray:
		pass

	@abstractmethod
	def fit(self, X: np.ndarray, y: np.ndarray):
		pass

	@abstractmethod
	def save(self, path: str):
		pass

	@classmethod
	@abstractmethod
	def load(cls, path: str) -> 'Model':
		pass

	def __call__(self, *args, **kwargs):
		return self.predict(*args, **kwargs)


class KerasModel(Model):

	def __init__(self, model: 'keras.Model'):
		self.__model = model

	def predict(self, inputs: np.ndarray) -> np.ndarray:
		return self.__model.predict(inputs)

	def fit(self, X: np.ndarray, y: np.ndarray):
		pass

	def save(self, path: str):
		pass

	@classmethod
	def load(cls, path: str) -> 'Model':
		pass


class TorchModel(Model):

	def __init__(self, model: nn.Module, device: torch.device = None):
		self.__model = model
		self.__model.eval()
		self.__device = None
		self.device = next(self.__model.parameters()).device if device is None else device

	@property
	def device(self) -> torch.device:
		return self.__device

	@device.setter
	def device(self, device: torch.device):
		Logger.info(f"Setting device: {device}")
		self.__device = device
		self.__model.to(device)

	def predict(self, inputs: np.ndarray) -> np.ndarray:
		self.__model.eval()
		self.__model.to(self.__device)
		with torch.no_grad():
			return self.__model(
				torch.from_numpy(inputs.astype(np.float32)).to(self.__device)
			).cpu().detach().numpy()

	def fit(self, X: np.ndarray, y: np.ndarray):
		pass

	def save(self, path: str):
		pass

	@classmethod
	def load(cls, path: str) -> 'Model':
		return TorchModel(ModelHandler.load(path))

