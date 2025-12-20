import typing

import torch

from core.utils.research.model.model.savable import SpinozaModule


class RevInNorm(SpinozaModule):

	def __init__(self, *args, encoder_channel: int = 0, **kwargs):
		self.args = {
			"encoder_channel": encoder_channel
		}
		super().__init__(*args, **kwargs)
		self.encoder_channel = encoder_channel

	def call(self, x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
		x = x[:, self.encoder_channel]
		return y * torch.std(x, dim=-1) + torch.mean(x, dim=-1)

	def export_config(self) -> typing.Dict[str, typing.Any]:
		return self.args
