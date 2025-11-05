import typing

import torch

from core.utils.research.model.model.savable import SpinozaModule
from .dynamic_layer_norm import DynamicLayerNorm


class NoiseInjectionLayer(SpinozaModule):

	def __init__(
			self,
			noise: float = 1e-3,
			frequency: float = 1.0,
			bounds: typing.Tuple[float, float] = (0.0, 1.0)
	):
		self.args = {
			"noise": noise,
			"frequency": frequency
		}
		super().__init__()
		self.noise = noise
		self.frequency = frequency
		self.norm = DynamicLayerNorm(elementwise_affine=False)
		self.bounds = bounds

	def apply_noise(self, x: torch.Tensor) -> torch.Tensor:
		bounds = int(x.shape[-1]*self.bounds[0]), int(x.shape[-1]*self.bounds[1])
		mask = torch.zeros_like(x)
		mask[..., bounds[0]:bounds[1]] = 1.0

		noise = self.noise * (
				self.norm(torch.randn_like(x))
		)
		return x + noise * mask

	def call(self, x: torch.Tensor) -> torch.Tensor:
		if not self.training:
			return x
		x = x.clone()
		sample_mask = torch.rand(x.size(0), device=x.device) <= self.frequency
		x[sample_mask] = self.apply_noise(x[sample_mask])
		return x

	def export_config(self) -> typing.Dict[str, typing.Any]:
		return self.args
