import typing

import torch
from torch import nn

from core.utils.research.data.prepare.smoothing_algorithm.lass.model.layers import EncoderNoiseInjectionLayer
from core.utils.research.model.model.cnn.collapse_block import CollapseBlock
from core.utils.research.model.model.savable import SpinozaModule
from core.utils.research.model.model.transformer import TransformerEmbeddingBlock, DecoderBlock
from .lass3_decoder_block import Lass3DecoderBlock
from .lass3_transformer_input_block import Lass3TransformerInputBlock


class Lass3Transformer(SpinozaModule):

	def __init__(
			self,
			block_size: int,
			encoder_embedding_block: TransformerEmbeddingBlock,
			decoder_embedding_block: TransformerEmbeddingBlock,
			encoder_block: DecoderBlock,
			decoder_block: Lass3DecoderBlock,
			collapse_block: CollapseBlock,
			input_block: Lass3TransformerInputBlock = None
	):
		self.args = {
			'block_size': block_size,
			'encoder_embedding_block': encoder_embedding_block,
			'decoder_embedding_block': decoder_embedding_block,
			'encoder_block': encoder_block,
			'decoder_block': decoder_block,
			'collapse_block': collapse_block,
			'input_block': input_block
		}
		super().__init__(input_size=(None, 2, block_size), auto_build=False)
		self.encoder_embedding_block = encoder_embedding_block
		self.decoder_embedding_block = decoder_embedding_block
		self.encoder_block = encoder_block
		self.decoder_block = decoder_block
		self.collapse_block = collapse_block

		self.input_block = input_block if input_block is not None else Lass3TransformerInputBlock()

		self.init()

	def call(self, x: torch.Tensor) -> torch.Tensor:
		x_encoder, x_decoder = self.input_block(x)

		x_encoder_embedded = self.encoder_embedding_block(x_encoder)
		y_encoder = self.encoder_block(x_encoder_embedded)

		x_decoder_embedded = self.decoder_embedding_block(x_decoder)
		y_decoder = self.decoder_block(y_encoder, x_decoder_embedded)

		y = self.collapse_block(y_decoder)
		return y

	def export_config(self) -> typing.Dict[str, typing.Any]:
		return self.args
