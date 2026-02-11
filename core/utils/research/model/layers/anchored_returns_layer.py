import typing

import torch
from torch import nn

from core.utils.research.model.layers import Log
from core.utils.research.model.model.savable import SpinozaModule


class AnchoredReturnsLayer(SpinozaModule):

	def __init__(
			self,
			anchored_channels: typing.List[typing.List[int]],
			anchor_channels: typing.List[int],
			log: bool = True
	):
		self.args = {
			"anchored_channels": anchored_channels,
			"anchor_channels": anchor_channels,
			"log": log
		}
		if isinstance(anchored_channels, typing.Sized) and not isinstance(anchored_channels[0], typing.Sized):
			anchored_channels = [anchored_channels]
		if isinstance(anchor_channels, int):
			anchor_channels = [anchor_channels]



		if len(anchored_channels) != len(anchor_channels):
			raise ValueError(f"number of anchored channels({len(anchored_channels)}) should be equal to number of anchor channels({len(anchor_channels)}")
		super().__init__()
		self.anchored_channels = anchored_channels
		self.anchor_channels = anchor_channels
		self.log = Log() if log else nn.Identity()

	def call(self, x: torch.Tensor) -> torch.Tensor:
		y = x[..., 1:].clone()
		for channels, anchor in zip(self.anchored_channels, self.anchor_channels):
			y[:, channels] = self.log(x[:, channels, 1:] / x[:, [anchor], :-1])

		return y

	def export_config(self) -> typing.Dict[str, typing.Any]:
		return self.args
