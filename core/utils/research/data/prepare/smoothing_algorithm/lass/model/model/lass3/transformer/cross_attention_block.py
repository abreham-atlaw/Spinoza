import torch
import torch.nn as nn

from core.utils.research.model.model.savable import SpinozaModule
from core.utils.research.model.model.transformer import DecoderBlock


class CrossAttentionBlock(DecoderBlock):

	def __init__(self, *args, **kwargs):
		super().__init__(*args, **kwargs)
		self.cross_attention_layer = None

	def cross_attention(self, x_encoder: torch.Tensor, x_decoder: torch.Tensor):
		if self.cross_attention_layer is None:
			self.cross_attention_layer = nn.MultiheadAttention(
				embed_dim=x_encoder.shape[-1],
				num_heads=self.num_heads,
				batch_first=True,
			)
		out, weights = self.cross_attention_layer(
			query=x_decoder,
			key=x_encoder,
			value=x_encoder
		)
		return out

	def _apply_attention(self, x: torch.Tensor, x_decoder: torch.Tensor = None):
		return self.cross_attention(x, x_decoder)
