import typing

import torch
import torch.nn as nn

from core.utils.research.model.model.savable import SpinozaModule


class TitanContextEmbeddingBlock(SpinozaModule):

	def __init__(
			self,
			instrument_positions: typing.Tuple[int, ...],
			instruments_vocab: int,
			embedding_size: int
	):
		self.args = {
			"instrument_positions": instrument_positions,
			"instruments_vocab": instruments_vocab,
			"embedding_size": embedding_size,
		}
		super().__init__()
		self.instrument_embedding = nn.Embedding(
			num_embeddings=instruments_vocab,
			embedding_dim=embedding_size,
		)
		self.instrument_positions = list(instrument_positions)

	def call(self, x: torch.Tensor) -> torch.Tensor:
		mask = torch.zeros(x.shape[-1], dtype=torch.bool)
		mask[self.instrument_positions] = True

		embedding = self.instrument_embedding(x[..., mask].to(torch.int))
		x = torch.concat([x[..., (~mask)], torch.flatten(embedding, start_dim=-2, end_dim=-1)], dim=-1)
		return x

	def export_config(self) -> typing.Dict[str, typing.Any]:
		return self.args
