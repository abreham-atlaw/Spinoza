import typing

import torch
import torch.nn as nn

from core.utils.research.model.layers import AddAndNorm
from core.utils.research.model.model.linear.model import LinearModel
from core.utils.research.model.model.savable import SpinozaModule
from core.utils.research.model.model.transformer.teb import TransformerEmbeddingBlock
from lib.utils.cache import Cache


class DecoderBlock(SpinozaModule):

	def __init__(
			self,
			num_heads: int,
			embedding_last: bool = False,
			norm_1: nn.Module = None,
			norm_2: nn.Module = None,
			ff_block: LinearModel = None
	):
		self.args = {
			'num_heads': num_heads,
			"embedding_last": embedding_last,
			"norm_1": norm_1,
			"norm_2": norm_2,
			"ff_block": ff_block
		}
		super().__init__()
		self.self_attention_layer = None
		self.num_heads = num_heads
		self.embedding_last = embedding_last
		self.norm_1 = AddAndNorm(norm_layer=norm_1) if norm_1 is not None else nn.Identity()
		self.norm_2 = AddAndNorm(norm_layer=norm_2) if norm_2 is not None else nn.Identity()
		self.ff_block = ff_block if ff_block is not None else nn.Identity()
		self.cache = Cache()

	def self_attention(self, x: torch.Tensor) -> torch.Tensor:
		if self.self_attention_layer is None:
			self.self_attention_layer = nn.MultiheadAttention(
				embed_dim=x.shape[-1],
				num_heads=self.num_heads,
				batch_first=True,
			)
		out, weights = self.self_attention_layer(x, x, x)
		return out

	def _apply_attention(self, x: torch.Tensor, x_decoder: torch.Tensor = None):
		return self.self_attention(x)

	def call(self, x: torch.Tensor, x_decoder: torch.Tensor = None) -> torch.Tensor:

		attention = self._apply_attention(x, x_decoder)
		attention = self.norm_1(x, attention)

		out = self.ff_block(attention)
		out = self.norm_2(attention, out) if isinstance(self.norm_2, AddAndNorm) else self.norm_2(out)

		if self.embedding_last:
			out = torch.transpose(out, 1, 2)
		return out

	def export_config(self) -> typing.Dict[str, typing.Any]:
		return self.args
