import typing

import torch
import torch.nn as nn

from core.utils.research.model.layers import FlattenLayer, Axis
from core.utils.research.model.model.linear.model import LinearModel
from core.utils.research.model.model.savable import SpinozaModule


class CollapseBlock(SpinozaModule):

	def __init__(
		self,
		ff_block: LinearModel = None,
		channel_ff_block: LinearModel = None,
		dropout: float = 0,
		extra_mode: bool = True,
		input_norm: typing.Optional[nn.Module] = None,
		global_avg_pool: bool = False
	):
		super().__init__(auto_build=False)
		self.args = {
			'ff_block': ff_block,
			"channel_ff_block": channel_ff_block,
			'dropout': dropout,
			"extra_mode": extra_mode,
			"input_norm": input_norm,
			"global_avg_pool": global_avg_pool
		}
		self.ff_block = ff_block
		self.channel_ff_block = Axis(channel_ff_block, axis=1) if channel_ff_block is not None else nn.Identity()
		self.dropout = nn.Dropout(dropout) if dropout > 0 else nn.Identity()
		self.extra_mode = extra_mode
		self.flatten = FlattenLayer(1, 2)
		self.input_norm = input_norm if input_norm is not None else nn.Identity()
		self.global_avg_pool = nn.AdaptiveAvgPool1d(1) if global_avg_pool else nn.Identity()

	def call(self, x: torch.Tensor, extra: typing.Optional[torch.Tensor] = None) -> torch.Tensor:
		normed = self.input_norm(x)

		normed = self.global_avg_pool(normed)

		channeled = self.channel_ff_block(normed)

		flattened = self.flatten(channeled)
		flattened = flattened.reshape(flattened.size(0), -1)
		flattened = self.dropout(flattened)

		concat = flattened
		if self.extra_mode:
			concat = torch.cat((concat, extra), dim=-1)

		out = self.ff_block(concat)
		return out

	def export_config(self) -> typing.Dict[str, typing.Any]:
		return self.args
