import typing

import torch
import torch.nn as nn

from core.utils.research.model.model.savable import SpinozaModule


class Lass3DeltaModel(SpinozaModule):

	def __init__(self, model: SpinozaModule):
		self.args = {
			"model": model
		}
		super().__init__(input_size=model.input_size, auto_build=False)
		self.model = model
		self.init()

	def call(self, x: torch.Tensor) -> torch.Tensor:
		recent_value = x[:, 1, -1:]
		y = self.model(x)
		return recent_value + y

	def export_config(self) -> typing.Dict[str, typing.Any]:
		return self.args
