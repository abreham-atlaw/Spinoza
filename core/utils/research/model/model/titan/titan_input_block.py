import typing

import torch

from core.utils.research.model.model.savable import SpinozaModule


class TitanInputBlock(SpinozaModule):

	def __init__(self, context_data_size: int):
		self.args = {
			"context_data_size": context_data_size,
		}
		super().__init__()
		self.context_data_size = context_data_size

	def call(self, x: torch.Tensor) -> torch.Tensor:
		time_series, context_data = x[..., :x.shape[-1] - self.context_data_size], x[..., x.shape[-1] - self.context_data_size:]
		return time_series, context_data

	def export_config(self) -> typing.Dict[str, typing.Any]:
		return self.args
