import typing

import torch
from torch import nn

from core.utils.research.model.layers import DynamicLayerNorm
from core.utils.research.model.model.savable import SpinozaModule


class EncoderNoiseInjectionLayer(SpinozaModule):

	def __init__(self, noise: float = 1e-4, frequency: float = 0.5, left_align: bool = False):
		self.args = {
			"noise": noise,
			"left_align": left_align
		}
		super().__init__()
		self.noise = noise
		self.left_align = left_align
		self.norm = DynamicLayerNorm(elementwise_affine=False)
		self.frequency = frequency

	def set_noise(self, noise: float):
		self.noise = noise

	def apply_noise(self, x_encoder: torch.Tensor, x_decoder: torch.Tensor) -> torch.Tensor:
		mask = torch.sign(x_decoder)
		if not self.left_align:
			mask = torch.flip(mask, dims=(-1,))

		noise = self.noise * (
					self.norm(torch.randn_like(x_encoder))
		) * mask
		return x_encoder + noise

	def call(self, x_encoder: torch.Tensor, x_decoder: torch.Tensor) -> torch.Tensor:
		if not self.training:
			return x_encoder
		x_encoder = x_encoder.clone()
		sample_mask = torch.rand(x_encoder.size(0), device=x_encoder.device) <= self.frequency
		x_encoder[sample_mask] = self.apply_noise(x_encoder[sample_mask], x_decoder[sample_mask])
		return x_encoder

	def export_config(self) -> typing.Dict[str, typing.Any]:
		return self.args
