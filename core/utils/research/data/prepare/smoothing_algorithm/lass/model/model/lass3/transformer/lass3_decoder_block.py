import typing

import torch

from core.utils.research.model.model.savable import SpinozaModule
from core.utils.research.model.model.transformer import DecoderBlock
from .cross_attention_block import CrossAttentionBlock


class Lass3DecoderBlock(SpinozaModule):

	def __init__(
			self,
			self_attention_block: DecoderBlock,
			cross_attention_block: CrossAttentionBlock,
			transpose_output: bool = False
	):
		super().__init__()
		self.args = {
			'self_attention_block': self_attention_block,
			'cross_attention_block': cross_attention_block,
			'transpose_output': transpose_output
		}
		self.self_attention_block = self_attention_block
		self.cross_attention_block = cross_attention_block
		self.transpose_output = transpose_output

	def call(self, x_encoder: torch.Tensor, x_decoder: torch.Tensor) -> torch.Tensor:

		self_attention = self.self_attention_block(x_decoder)
		out = self.cross_attention_block(x_encoder, self_attention)

		if self.transpose_output:
			out = torch.transpose(out, 1, 2)

		return out

	def export_config(self) -> typing.Dict[str, typing.Any]:
		return self.args
