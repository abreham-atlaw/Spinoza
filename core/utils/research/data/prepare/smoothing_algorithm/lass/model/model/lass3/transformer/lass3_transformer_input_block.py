import typing

import torch
import torch.nn as nn

from core.utils.research.data.prepare.smoothing_algorithm.lass.model.layers import EncoderNoiseInjectionLayer
from core.utils.research.model.model.savable import SpinozaModule


class Lass3TransformerInputBlock(SpinozaModule):

	def __init__(
			self,
			encoder_prep: nn.Module = None,
			decoder_prep: nn.Module = None,
			*args,
			**kwargs
	):
		self.args = {
			"encoder_prep": encoder_prep,
			"decoder_prep": decoder_prep
		}
		super().__init__()
		self.encoder_prep_layer = encoder_prep
		self.decoder_prep_layer = decoder_prep if decoder_prep is not None else nn.Identity()

	def prep_encoder(self, x_encoder: torch.Tensor, x_decoder: torch.Tensor) -> torch.Tensor:
		if self.encoder_prep_layer is None:
			return x_encoder
		return self.encoder_prep_layer(x_encoder, x_decoder)

	def call(self, x: torch.Tensor) -> typing.Tuple[torch.Tensor, torch.Tensor]:
		x_encoder, x_decoder = x[:, 0, :], x[:, 1, :]
		x_encoder = self.prep_encoder(x_encoder, x_decoder)

		x_decoder = self.decoder_prep_layer(x_decoder)
		return x_encoder, x_decoder

	def export_config(self) -> typing.Dict[str, typing.Any]:
		return self.args
