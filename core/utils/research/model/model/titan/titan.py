import typing

import torch

from core.utils.research.model.model.savable import SpinozaModule
from .titan_time_series_block import TitanTimeSeriesBlock
from .titan_context_block import TitanContextBlock
from .titan_input_block import TitanInputBlock

class TitanModel(SpinozaModule):

	def __init__(
			self,
			input_size: typing.Tuple[int, int],
			input_block: TitanInputBlock,
			time_series_block: TitanTimeSeriesBlock,
			context_block: TitanContextBlock,
	):
		self.args = {
			"input_size": input_size,
			"input_block": input_block,
			"time_series_block": time_series_block,
			"context_block": context_block,
		}
		super().__init__(input_size=(None,)+tuple(input_size), auto_build=False)
		self.input_block = input_block
		self.time_series_block = time_series_block
		self.context_block = context_block
		self.init()

	def call(self, x:torch.Tensor) -> torch.Tensor:
		time_series, context = self.input_block(x)
		time_series_output, time_series_latent = self.time_series_block(time_series)
		context_output = self.context_block(x, time_series_latent)

		y = torch.zeros((x.shape[0], time_series_output.shape[1], time_series_output.shape[2]+1))
		y[..., :-1] = time_series_output
		y[:, 0] = context_output

		return y

	def export_config(self) -> typing.Dict[str, typing.Any]:
		return self.args
