import typing

import torch
import torch.nn as nn

from core.utils.research.model.model.savable import SpinozaModule


class DecoderPadEmbedding(SpinozaModule):

	def __init__(self):
		super().__init__()
		self.args = {}
		self.pad_embedding = nn.Parameter(torch.zeros(1,))

	def call(self, x: torch.Tensor) -> torch.Tensor:
		x = torch.where(x == 0, self.pad_embedding.expand_as(x), x)
		return x

	def export_config(self) -> typing.Dict[str, typing.Any]:
		return self.args
