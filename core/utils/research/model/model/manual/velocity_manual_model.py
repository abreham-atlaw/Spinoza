import typing

import torch

from core.utils.research.data.prepare.utils.data_prep_utils import DataPrepUtils
from core.utils.research.model.model.savable import SpinozaModule


class VelocityManualModel(SpinozaModule):



	def __init__(
			self,
			extra_len: int,
			bounds: typing.List[float],
			y_extra_len: int = 1,
	):
		self.args = {
			"extra_len": extra_len,
			"bounds": bounds
		}
		super(VelocityManualModel, self).__init__()
		self.bounds = torch.Tensor(bounds)
		self.extra_len = extra_len
		self.vocab_size = self.bounds.shape[0] + 1
		self.y_extra_len = y_extra_len

	def call(self, x: torch.Tensor) -> torch.Tensor:
		x = x[..., :x.shape[-1] - self.extra_len]
		v = torch.unsqueeze(x[..., -1] / x[..., -2], dim=-1)

		y = torch.zeros((x.shape[0], self.vocab_size))
		y[torch.arange(x.shape[0]), torch.sum(v >= self.bounds, dim=-1)] = 1

		y = torch.concatenate(
			(y, torch.zeros((y.shape[0], self.y_extra_len))),
			dim=-1
		)

		return y

	def export_config(self) -> typing.Dict[str, typing.Any]:
		return self.args
