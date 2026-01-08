import typing

import torch
import torch.nn as nn
import torch.nn.functional as F

from core.utils.research.model.model.savable import SpinozaModule
from lib.utils.logger import Logger


class HorizonModel(SpinozaModule):

	def __init__(
			self,
			h: float,
			bounds: typing.Union[typing.List[float], torch.Tensor],
			model: SpinozaModule,
			X_extra_len: int = 124,
			y_extra_len: int = 1,
			max_depth: int = None,
			use_gumbel_softmax: bool = False,
			gumbel_softmax_temperature: float = 0.1
	):
		self.args = {
			"h": h,
			"bounds": bounds,
			"model": model,
			"X_extra_len": X_extra_len,
			"y_extra_len": y_extra_len,
			"max_depth": max_depth,
			"use_gumbel_softmax": use_gumbel_softmax,
			"gumbel_softmax_temperature": gumbel_softmax_temperature
		}
		super().__init__(input_size=model.input_size, output_size=model.output_size, auto_build=False)
		Logger.info(f"Initializing HorizonModel(h={h}, max_depth={max_depth})...")
		self.h = h
		self.model = model
		self.X_extra_len = X_extra_len
		self.y_extra_len = y_extra_len
		self.softmax = nn.Softmax(dim=-1)

		self.bounds = self.__prepare_bounds(bounds)
		self.__max_depth = max_depth

		self.use_gumbel_softmax = use_gumbel_softmax
		self.gumbel_softmax_temperature = gumbel_softmax_temperature

	def set_h(self, h: float):
		self.h = h

	def __prepare_bounds(self, bounds: typing.Union[typing.List[float], torch.Tensor]) -> torch.Tensor:
		if isinstance(bounds, typing.List):
			bounds = torch.tensor(bounds)

		epsilon = (bounds[1] - bounds[0] +  bounds[-1] - bounds[-2])/2
		Logger.info(f"Using epsilon: {epsilon}")
		bounds = torch.cat([
			torch.Tensor([bounds[0] - epsilon]),
			bounds,
			torch.Tensor([bounds[-1] + epsilon])
		])
		bounds = (bounds[1:] + bounds[:-1])/2
		self.register_buffer("bounds", bounds)
		return bounds

	def __check_depth(self, depth: int) -> bool:
		if depth is None or self.__max_depth is None:
			return True
		return depth < self.__max_depth

	def _retrieve_recent_close(self, x: torch.Tensor):
		return  x[..., -(self.X_extra_len + 1)]

	def _predict(self, x: torch.Tensor, depth: int) -> torch.Tensor:
		y = self(x, depth+1)[..., :-self.y_extra_len]

		if self.use_gumbel_softmax:
			return F.gumbel_softmax(y, tau=self.gumbel_softmax_temperature, hard=False, dim=-1)

		return self.softmax(y)

	def shift_and_predict(self, x: torch.Tensor, depth: int) -> torch.Tensor:
		x[..., 1:x.shape[-1]-self.X_extra_len] = x[..., 0:-(self.X_extra_len + 1)].clone()
		y = self._predict(x, depth)
		if y.ndim == 2:
			y = torch.unsqueeze(y, dim=1)
		y = torch.sum(
			y*self.bounds,
			dim=-1
		) * self._retrieve_recent_close(x)
		return y

	def process_sample(self, x: torch.Tensor, depth: int) -> torch.Tensor:
		x[..., -(self.X_extra_len + 1)] = self.shift_and_predict(x.clone(), depth)
		return x

	def call(self, x: torch.Tensor, depth: int = 0) -> torch.Tensor:
		x = x.clone()
		sample_mask = torch.rand(x.size(0)) <= self.h

		if self.__check_depth(depth) and torch.any(sample_mask):
			x[sample_mask] = self.process_sample(x[sample_mask], depth)

		return self.model(x)

	def export_config(self) -> typing.Dict[str, typing.Any]:
		Logger.warning("Exporting HorizonModel. This is not recommended as it should be used as a wrapper. Use HorizonModel.model instead.")
		return self.args
