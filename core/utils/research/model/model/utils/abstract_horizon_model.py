import typing
from abc import ABC, abstractmethod

import torch

from core.utils.research.model.model.savable import SpinozaModule
from lib.utils.logger import Logger


class AbstractHorizonModel(SpinozaModule, ABC):

	def __init__(
			self,
			h: float,
			model: SpinozaModule,
			max_depth: int = None
	):
		self.args = {
			"h": h,
			"model": model,
			"max_depth": max_depth
		}
		super().__init__(input_size=model.input_size, output_size=model.output_size, auto_build=False)
		Logger.info(f"Initializing HorizonModel(h={h}, max_depth={max_depth})...")
		self.h = h
		self.model = model
		self.max_depth = max_depth

	def set_h(self, h: float):
		Logger.info(f"Setting h to {h}")
		self.h = h

	def _check_depth(self, depth: int, x: torch.Tensor) -> bool:
		return self.max_depth is None or depth < self.max_depth

	@abstractmethod
	def _shift(self, x: torch.Tensor) -> torch.Tensor:
		pass

	@abstractmethod
	def _place_prediction(self, x:torch.Tensor, y:torch.Tensor) -> torch.Tensor:
		pass

	def _predict_shifted(self, x: torch.Tensor, depth: int) -> torch.Tensor:
		return self(x, depth+1)

	def _shift_and_predict(self, x: torch.Tensor, depth: int) -> torch.Tensor:
		x = self._shift(x.clone())
		return self._predict_shifted(x, depth)

	def _process_sample(self, x: torch.Tensor, depth: int) -> torch.Tensor:
		prediction = self._shift_and_predict(x.clone(), depth)
		x = self._place_prediction(x, prediction)
		return x

	def call(self, x: torch.Tensor, depth: int = 0) -> torch.Tensor:

		x = x.clone()
		sample_mask = torch.rand(x.size(0)) <= self.h

		if self._check_depth(depth, x[sample_mask]) and torch.any(sample_mask):
			x[sample_mask] = self._process_sample(x[sample_mask], depth)

		return self.model(x)

	def export_config(self) -> typing.Dict[str, typing.Any]:
		Logger.warning(
			"Exporting HorizonModel. This is not recommended as it should be used as a wrapper. Use HorizonModel.model instead."
		)
		return self.args
