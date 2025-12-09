import typing

import torch
import torch.nn as nn

from core.utils.research.model.layers import FlattenLayer
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
		global_avg_pool: bool = False,
		flatten: bool = True
	):
		super().__init__(auto_build=False)
		self.args = {
			'ff_block': ff_block,
			"channel_ff_block": channel_ff_block,
			'dropout': dropout,
			"extra_mode": extra_mode,
			"input_norm": input_norm,
			"global_avg_pool": global_avg_pool,
			"flatten": flatten
		}
		self.ff_block = ff_block
		self.channel_ff_block = channel_ff_block if channel_ff_block is not None else nn.Identity()
		self.dropout = nn.Dropout(dropout) if dropout > 0 else nn.Identity()
		self.extra_mode = extra_mode
		self.flatten = FlattenLayer(1, 2) if flatten else nn.Identity()
		self.input_norm = input_norm if input_norm is not None else nn.Identity()
		self.global_avg_pool = nn.AdaptiveAvgPool1d(1) if global_avg_pool else nn.Identity()

	def call(self, x: torch.Tensor, extra: typing.Optional[torch.Tensor] = None) -> torch.Tensor:
		normed = self.input_norm(x)

		normed = self.global_avg_pool(normed)

		channeled = self.channel_ff_block(normed)

		flattened = self.flatten(channeled)
		flattened = self.dropout(flattened)

		concat = flattened
		if self.extra_mode:
			concat = torch.cat((concat, extra), dim=-1)

		out = self.ff_block(concat)
		return out

	def export_config(self) -> typing.Dict[str, typing.Any]:
		return self.args
