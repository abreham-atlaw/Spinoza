import typing

import torch
import torch.nn as nn

from core.utils.research.model.layers import ReverseSoftmax
from core.utils.research.model.model.savable import SpinozaModule
from lib.utils.logger import Logger


class AggregateModel(SpinozaModule):

	def __init__(
			self,
			model: SpinozaModule,
			bounds: typing.Union[typing.List[float], torch.Tensor],
			a: float,
			y_extra_len: int = 1,
			temperature: float = 1e-5,
			softmax: bool = False
	):

		self.args = {
			"model": model,
			"bounds": bounds,
			"a": a,
			"y_extra_len": y_extra_len,
			"temperature": temperature,
			"softmax": softmax
		}
		super().__init__(
			input_size=model.input_size if isinstance(model, SpinozaModule) else None,
			auto_build=False
		)
		self.model = model
		self.raw_bounds, self.bounds = self.__prepare_bounds(bounds)
		self.a = a
		self.n = int(1.0 / a)
		self.y_extra_len = y_extra_len
		self.temperature = temperature
		self.temperature_softmax = nn.Softmax(dim=-1)

		self.softmax, self.reverse_softmax = (nn.Softmax(dim=-1), ReverseSoftmax(dim=-1)) if softmax else (nn.Identity(), nn.Identity())

		Logger.info(f"Initializing AggregateModel with a={a}, softmax={softmax}")

	def __prepare_bounds(self, bounds: typing.Union[typing.List[float], torch.Tensor]) -> torch.Tensor:
		if isinstance(bounds, typing.List):
			bounds = torch.tensor(bounds)

		raw_bounds = torch.clone(bounds)

		epsilon = (bounds[1] - bounds[0] +  bounds[-1] - bounds[-2])/2
		Logger.info(f"Using epsilon: {epsilon}")
		bounds = torch.cat([
			torch.Tensor([bounds[0] - epsilon]),
			bounds,
			torch.Tensor([bounds[-1] + epsilon])
		])

		bounds = (bounds[1:] + bounds[:-1])/2

		self.register_buffer("raw_bounds", bounds)
		self.register_buffer("bounds", bounds)
		return raw_bounds, bounds

	def norm(self, x: torch.Tensor) -> torch.Tensor:
		return x / torch.sum(x, dim=-1, keepdim=True)

	def select_and_aggregate(self, x: torch.Tensor) -> torch.Tensor:
		p = torch.sum(
			x[torch.reshape(torch.arange(x.shape[0]), (-1, 1)), torch.flip(torch.argsort(x), dims=[1])[:, :self.n]],
			dim=-1
		)

		samples_mask = p < self.n * self.a

		if torch.any(samples_mask):
			x[samples_mask] = self.aggregate(x[samples_mask])

		return x

	def __apply_certainty(self, x: torch.Tensor) -> torch.Tensor:
		return self.temperature_softmax(x / self.temperature)

	def aggregate(self, x: torch.Tensor) -> torch.Tensor:

		selection_mask = x < self.a

		idx_1 = torch.flatten(torch.multinomial(self.__apply_certainty(x * selection_mask), num_samples=1))
		idx_2 = torch.flatten(torch.multinomial(
			self.__apply_certainty(self.norm(torch.nan_to_num(x / torch.abs(self.bounds - torch.reshape(self.bounds[idx_1], (-1, 1))), posinf=0.0)) * selection_mask),
			num_samples=1
		))

		y = torch.clone(x)
		y[torch.arange(y.shape[0]), idx_1] = 0
		y[torch.arange(y.shape[0]), idx_2] = 0

		v = (
			(self.bounds[idx_1]*x[torch.arange(y.shape[0]), idx_1] + self.bounds[idx_2]*x[torch.arange(y.shape[0]), idx_2]) /
			(x[torch.arange(y.shape[0]), idx_1] + x[torch.arange(y.shape[0]), idx_2])
		)

		y[torch.arange(y.shape[0]), torch.sum(torch.reshape(v, (-1, 1)) >= self.raw_bounds, dim=-1)] += x[torch.arange(y.shape[0]), idx_1]+ x[torch.arange(y.shape[0]), idx_2]

		return self.select_and_aggregate(y)

	def aggregate_y(self, y: torch.Tensor) -> torch.Tensor:
		if len(y.shape) == 3:
			return torch.stack([
				self.aggregate_y(y[:,i])
				for i in range(y.shape[1])
			], dim=1)

		return torch.concatenate(
			[
				self.reverse_softmax(self.aggregate(self.softmax(y[..., :y.shape[-1] - self.y_extra_len]))),
				y[..., y.shape[-1] - self.y_extra_len:]
			],
			dim=-1
		)

	def call(self, x: torch.Tensor) -> torch.Tensor:
		y = self.model(x)

		return self.aggregate_y(y)

	def export_config(self) -> typing.Dict[str, typing.Any]:
		return self.args
