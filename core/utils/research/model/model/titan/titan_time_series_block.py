import typing

import torch
import torch.nn as nn

from core.utils.research.model.model.cnn.bridge_block import BridgeBlock
from core.utils.research.model.model.cnn.cnn_block import CNNBlock
from core.utils.research.model.model.cnn.collapse_block import CollapseBlock
from core.utils.research.model.model.cnn.embedding_block import EmbeddingBlock
from core.utils.research.model.model.savable import SpinozaModule


class TitanTimeSeriesBlock(SpinozaModule):

	def __init__(
			self,
			embedding_block: EmbeddingBlock,
			cnn_block: CNNBlock,
			collapse_block: CollapseBlock,
			bridge_block: typing.Optional[BridgeBlock] = None
	):
		self.args = {
			"embedding_block": embedding_block,
			"cnn_block": cnn_block,
			"collapse_block": collapse_block,
			"bridge_block": bridge_block
		}
		super().__init__()
		self.embedding_block = embedding_block
		self.cnn_block = cnn_block
		self.collapse_block = collapse_block
		self.bridge_block = bridge_block if bridge_block is not None else nn.Identity()

	def call(self, x: torch.Tensor) -> torch.Tensor:
		embedded = self.embedding_block(x)
		cnn_out = self.cnn_block(embedded)
		latent = self.bridge_block(cnn_out)

		out = self.collapse_block(latent)

		return out, latent

	def export_config(self) -> typing.Dict[str, typing.Any]:
		return self.args
