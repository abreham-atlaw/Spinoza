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
			a: typing.Union[float, typing.List[float]],
			y_extra_len: int = 1,
			temperature: float = 1e-5,
			softmax: bool = False,
			masking_value: float = -1e9
	):

		self.args = {
			"model": model,
			"bounds": bounds,
			"a": a,
			"y_extra_len": y_extra_len,
			"temperature": temperature,
			"softmax": softmax,
			"masking_value": masking_value
		}
		super().__init__(
			input_size=model.input_size if isinstance(model, SpinozaModule) else None,
			auto_build=False
		)
		self.model = model
		self.raw_bounds, self.bounds = self.__prepare_bounds(bounds)
		self.a_list = a if isinstance(a, list) else [a]
		self.a = self.a_list[0]
		self.n_list = [int(1.0/alpha) for alpha in self.a_list]
		self.n = self.n_list[0]
		self.y_extra_len = y_extra_len
		self.temperature = temperature
		self.temperature_softmax = nn.Softmax(dim=-1)
		self.masking_value = masking_value

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

	def __activate_channel(self, i: int):
		if len(self.a_list) == 1:
			return
		self.a = self.a_list[i]
		self.n = self.n_list[i]

	def norm(self, x: torch.Tensor) -> torch.Tensor:
		return x / torch.sum(x, dim=-1, keepdim=True)

	def select_and_aggregate(self, x: torch.Tensor) -> torch.Tensor:
		p = torch.sum(
			x[torch.reshape(torch.arange(x.shape[0]), (-1, 1)), torch.flip(torch.argsort(x), dims=[1])[:, :self.n]],
			dim=-1
		)

		samples_mask = p < self.n * self.a

		if torch.any(samples_mask):
			x = x.clone()
			x[samples_mask] = self.aggregate(x[samples_mask])

		return x

	def __apply_certainty(self, x: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
		x = torch.where(mask, x, torch.tensor(self.masking_value))
		return self.temperature_softmax(x / self.temperature)

	def aggregate(self, x: torch.Tensor) -> torch.Tensor:

		selection_mask = x < self.a

		idx_1 = torch.flatten(torch.multinomial(self.__apply_certainty(x, selection_mask), num_samples=1))
		selection_mask[torch.arange(x.shape[0]), idx_1] = False
		idx_2 = torch.flatten(torch.multinomial(
			self.__apply_certainty(
				self.norm(torch.nan_to_num(x / torch.abs(self.bounds - torch.reshape(self.bounds[idx_1], (-1, 1))), posinf=0.0)),
				selection_mask
			),
			num_samples=1
		))

		y = torch.clone(x)
		y[torch.arange(y.shape[0]), idx_1] = 0
		y[torch.arange(y.shape[0]), idx_2] = 0

		v = (
			(self.bounds[idx_1]*x[torch.arange(y.shape[0]), idx_1] + self.bounds[idx_2]*x[torch.arange(y.shape[0]), idx_2]) /
			(x[torch.arange(y.shape[0]), idx_1] + x[torch.arange(y.shape[0]), idx_2])
		)

		y[torch.arange(y.shape[0]), torch.sum(torch.reshape(v, (-1, 1)) >= self.raw_bounds, dim=-1)] += x[torch.arange(y.shape[0]), idx_1] + x[torch.arange(y.shape[0]), idx_2]

		return self.select_and_aggregate(y)

	def aggregate_y(self, y: torch.Tensor) -> torch.Tensor:
		if len(y.shape) == 3:
			aggregated = []
			for i in range(y.shape[1]):
				self.__activate_channel(i)
				aggregated.append(self.aggregate_y(y[:,i]))
			return torch.stack(aggregated, dim=1)

		return torch.concatenate(
			[
				self.reverse_softmax(self.select_and_aggregate(self.softmax(y[..., :y.shape[-1] - self.y_extra_len]))),
				y[..., y.shape[-1] - self.y_extra_len:]
			],
			dim=-1
		)

	def call(self, x: torch.Tensor) -> torch.Tensor:
		y = self.model(x)

		return self.aggregate_y(y)

	def export_config(self) -> typing.Dict[str, typing.Any]:
		return self.args
