import typing

import torch
from torch import nn

from .decoded_encoder_abstract_layer import DecodedEncoderAbstractLayer


class DecodedEncoderDropout(DecodedEncoderAbstractLayer):

	def __init__(self, dropout: float, *args, **kwargs):
		super().__init__(*args, **kwargs)
		self.args.update({
			"dropout": dropout
		})
		self.dropout = dropout

	def _call(self, x_encoder: torch.Tensor, x_decoder: torch.Tensor, decoded_mask: torch.Tensor) -> torch.Tensor:
		if not self.training or self.dropout == 0:
			return x_encoder

		sample_mask = torch.rand(x_encoder.size(0), device=x_encoder.device) <= self.dropout
		x_encoder[sample_mask.reshape((-1, 1)) & decoded_mask] = 0.0
		return x_encoder
