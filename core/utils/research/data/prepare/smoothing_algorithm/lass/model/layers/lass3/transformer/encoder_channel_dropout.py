import typing

import torch

from core.utils.research.model.model.savable import SpinozaModule


class EncoderChannelDropout(SpinozaModule):

	def __init__(self, dropout: float, *args, **kwargs):
		self.args = {
			"dropout": dropout
		}
		super().__init__(*args, **kwargs)
		self.dropout = dropout

	def call(self, x_encoder: torch.Tensor, x_decoder: torch.Tensor) -> torch.Tensor:
		if not self.training or self.dropout == 0:
			return x_encoder

		sample_mask = torch.rand(x_encoder.size(0), device=x_encoder.device) <= self.dropout
		x_encoder[sample_mask] = 0.0
		return x_encoder

	def export_config(self) -> typing.Dict[str, typing.Any]:
		return self.args
