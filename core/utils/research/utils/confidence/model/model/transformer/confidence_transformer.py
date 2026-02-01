import typing

import torch

from core.utils.research.data.prepare.smoothing_algorithm.lass.model.model.lass3.transformer import CrossAttentionBlock
from core.utils.research.model.model.cnn.collapse_block import CollapseBlock
from core.utils.research.model.model.savable import SpinozaModule
from core.utils.research.model.model.transformer import DecoderBlock, TransformerEmbeddingBlock
from core.utils.research.utils.confidence.model.model.transformer import ConfidenceTransformerInputBlock


class ConfidenceTransformer(SpinozaModule):

	def __init__(
			self,
			input_shape: typing.Tuple[int, int],
			encoder_embedding_block: TransformerEmbeddingBlock,
			decoder_embedding_block: TransformerEmbeddingBlock,
			encoder_block: DecoderBlock,
			decoder_block: DecoderBlock,
			cross_attention_block: CrossAttentionBlock,
			collapse_block: CollapseBlock,
			input_block: ConfidenceTransformerInputBlock = None,
	):
		self.args = {
			'input_shape': input_shape,
			'encoder_embedding_block': encoder_embedding_block,
			'decoder_embedding_block': decoder_embedding_block,
			'encoder_block': encoder_block,
			'decoder_block': decoder_block,
			'cross_attention_block': cross_attention_block,
			'collapse_block': collapse_block,
			'input_block': input_block,
		}
		super().__init__(input_size=(None, )+tuple(input_shape), auto_build=False)
		self.encoder_embedding_block = encoder_embedding_block
		self.decoder_embedding_block = decoder_embedding_block
		self.encoder_block = encoder_block
		self.decoder_block = decoder_block
		self.cross_attention_block = cross_attention_block
		self.collapse_block = collapse_block
		self.input_block = input_block if input_block is not None else ConfidenceTransformerInputBlock()
		self.init()

	def call(self, x: torch.Tensor) -> torch.Tensor:
		x_encoder, x_decoder = self.input_block(x)

		x_encoder_embedded = self.encoder_embedding_block(x_encoder)
		x_decoder_embedded = self.decoder_embedding_block(x_decoder)

		y_encoder = self.encoder_block(x_encoder_embedded)
		y_decoder = self.decoder_block(x_decoder_embedded)

		cross_attention = self.cross_attention_block(y_encoder, y_decoder)

		cross_attention = torch.transpose(cross_attention, 1, 2)

		y = self.collapse_block(cross_attention)

		return y

	def export_config(self) -> typing.Dict[str, typing.Any]:
		return self.args
