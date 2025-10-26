import typing

import torch

from core.utils.research.model.model.savable import SpinozaModule


class EncoderVerticalShift(SpinozaModule):

	def __init__(self, shift: float = 1.0):
		self.args = {
			"shift": shift
		}
		super().__init__()

	def call(self, x_encoder: torch.Tensor, x_decoder: torch.Tensor) -> torch.Tensor:
		if not self.training:
			return x_encoder
		x_encoder += (x_encoder[:, :1] * (torch.rand_like(x_encoder[:, :1]) - 0.5))
		return x_encoder

	def export_config(self) -> typing.Dict[str, typing.Any]:
		return self.args
