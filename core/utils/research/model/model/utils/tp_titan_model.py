import typing

import torch

from core.utils.research.model.model.savable import SpinozaModule


class TPTitanModel(SpinozaModule):

	def __init__(
			self,
			model: SpinozaModule,
			extra_len: int
	):
		self.args = {
			"model": model,
			"extra_len": extra_len
		}
		super().__init__(input_size=(None, model.input_size[1], model.input_size[2] + extra_len), auto_build=False)
		self.model = model
		self.extra_len = extra_len
		self.init()

	def call(self, x: torch.Tensor) -> torch.Tensor:
		x = x[..., :x.shape[-1] - self.extra_len]
		y_hat = self.model(x)
		return y_hat

	def export_config(self) -> typing.Dict[str, typing.Any]:
		return self.args

