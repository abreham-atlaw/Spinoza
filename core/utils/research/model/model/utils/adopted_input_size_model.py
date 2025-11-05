import typing

import torch

from core.utils.research.model.model.savable import SpinozaModule


class AdoptedInputSizeModel(SpinozaModule):

	def __init__(
			self,
			input_size: int,
			model: SpinozaModule,
			*args,
			**kwargs
	):
		self.args = {
			"input_size": input_size,
			"model": model
		}
		super().__init__(*args, input_size=input_size, auto_build=False, **kwargs)
		self.model = model
		self.init()

	def call(self, x: torch.Tensor) -> torch.Tensor:
		x = x[:, -self.model.input_size[1]:]
		return self.model(x)

	def export_config(self) -> typing.Dict[str, typing.Any]:
		return self.args
