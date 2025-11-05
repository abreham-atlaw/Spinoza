import json
import typing
from abc import ABC, abstractmethod

import torch
import torch.nn as nn

from lib.utils.logger import Logger


class SpinozaLoss(nn.Module, ABC):

	def __init__(
			self,
			*args,
			weighted_sample: bool = False,
			collapsed: bool = True,
			**kwargs
	):
		super().__init__(*args, **kwargs)
		self.weighted_sample = weighted_sample
		self.collapsed = collapsed

	@abstractmethod
	def _call(self, y_hat: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
		pass

	def _collapse(self, loss: torch.Tensor) -> torch.Tensor:
		return torch.mean(loss)

	def _apply_weights(self, loss: torch.Tensor, w: torch.Tensor) -> torch.Tensor:
		return loss * w

	def forward(self, y_hat: torch.Tensor, y: torch.Tensor, w: torch.Tensor = None) -> torch.Tensor:
		loss = self._call(y_hat, y)

		if self.weighted_sample:
			loss = self._apply_weights(loss, w)

		if self.collapsed:
			loss = self._collapse(loss)

		return loss

	def __str__(self):
		return f"{self.__class__.__name__}(weighted_sample={self.weighted_sample}, collapsed={self.collapsed})"

	def _export_configs(self) -> typing.Dict[str, typing.Any]:
		return {
			"weighted_sample": self.weighted_sample,
			"collapsed": self.collapsed
		}

	def save(self, path: str):
		Logger.info(f"Saving {self} to \"{path}\"")
		configs = self._export_configs()
		with open(path, "w") as file:
			json.dump(configs, file)

	@classmethod
	def _import_configs(cls, configs: typing.Dict[str, typing.Any]) -> typing.Dict[str, typing.Any]:
		return configs

	@classmethod
	def load(cls, path: str, *args, **kwargs) -> 'SpinozaLoss':
		Logger.info(f"Loading {cls} from \"{path}\"")
		with open(path, "r") as file:
			configs = json.load(file)
		configs = cls._import_configs(configs)
		return cls(*args, **kwargs, **configs)
