import typing

import torch

from core.utils.research.model.layers import FlattenLayer
from core.utils.research.model.model.linear.model import LinearModel
from core.utils.research.model.model.savable import SpinozaModule
from core.utils.research.model.model.titan import TitanContextEmbeddingBlock
from core.utils.research.model.model.transformer import DecoderBlock, TransformerEmbeddingBlock


class TitanContextBlock(SpinozaModule):

	def __init__(
			self,
			embedding_block: TitanContextEmbeddingBlock,
			context_ffn: LinearModel,
			time_series_decoder_block: DecoderBlock,
			time_series_ffn: LinearModel,
			value_head: LinearModel
	):
		self.args = {
			"embedding_block": embedding_block,
			"context_ffn": context_ffn,
			"time_series_decoder_block": time_series_decoder_block,
			"time_series_ffn": time_series_ffn,
			"value_head": value_head
		}
		super().__init__()
		self.embedding_block = embedding_block
		self.context_ffn = context_ffn
		self.time_series_input_block = TransformerEmbeddingBlock()
		self.time_series_decoder_block = time_series_decoder_block
		self.time_series_ffn = time_series_ffn
		self.value_head = value_head
		self.time_series_flatten = FlattenLayer(-2, -1)

	def call(self, x: torch.Tensor, time_series_latent: torch.Tensor) -> torch.Tensor:
		x = x[:, 0]

		embedded = self.embedding_block(x)
		context_latent = self.context_ffn(embedded)

		attn_in = self.time_series_input_block(time_series_latent)
		attn_out = self.time_series_decoder_block(attn_in)
		attn_ffn_out = self.time_series_ffn(attn_out)
		attn_ffn_out = self.time_series_flatten(attn_ffn_out)

		concat = torch.concatenate([context_latent, attn_ffn_out], dim=1)
		value = self.value_head(concat)

		return value

	def export_config(self) -> typing.Dict[str, typing.Any]:
		return self.args
