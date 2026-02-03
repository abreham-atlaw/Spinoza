import typing

import torch

from core.utils.research.model.model.savable import SpinozaModule


class ConfidenceTransformerInputBlock(SpinozaModule):

	def __init__(self, seq_len: int = 128):
		self.args = {
			"seq_len": seq_len,
		}
		super().__init__()
		self.seq_len = seq_len

	def call(self, x: torch.Tensor) -> torch.Tensor:
		return x[..., :self.seq_len], x[..., self.seq_len:]

	def export_config(self) -> typing.Dict[str, typing.Any]:
		return self.args
