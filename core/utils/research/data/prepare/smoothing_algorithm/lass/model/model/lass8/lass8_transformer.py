import typing

import torch

from core.utils.research.model.model.cnn.collapse_block import CollapseBlock
from core.utils.research.model.model.savable import SpinozaModule
from core.utils.research.model.model.transformer import TransformerEmbeddingBlock, DecoderBlock


class Lass8Transformer(SpinozaModule):

	def __init__(
			self,
			block_size: int,
			embedding_block: TransformerEmbeddingBlock,
			encoder_block: DecoderBlock,
			decoder_block: DecoderBlock,
			collapse_block: CollapseBlock,
	):
		self.args = {
			"block_size": block_size,
			"embedding_block": embedding_block,
			"encoder_block": encoder_block,
			"decoder_block": decoder_block,
			"collapse_block": collapse_block,
		}
		super().__init__(
			input_size=block_size,
			auto_build=False
		)
		self.embedding_block = embedding_block
		self.encoder_block = encoder_block
		self.decoder_block = decoder_block
		self.collapse_block = collapse_block
		self.init()

	def call(self, x: torch.Tensor) -> torch.Tensor:

		embedded = self.embedding_block(x)
		encoded = self.encoder_block(embedded)
		decoded = self.decoder_block(encoded)

		decoded = torch.transpose(decoded, 1, 2)

		y = self.collapse_block(decoded)

		return y

	def export_config(self) -> typing.Dict[str, typing.Any]:
		return self.args