import typing

import torch
from torch import nn

from core.utils.research.model.model.savable import SpinozaModule


class WrappedConfidenceModel(SpinozaModule):

	def __init__(self, core_model: SpinozaModule, confidence_model: SpinozaModule):
		self.args = {
			"core_model": core_model,
			"confidence_model": confidence_model
		}
		super().__init__(input_size=core_model.input_size, auto_build=False)
		self.core_model = core_model
		self.confidence_model = confidence_model
		self.init()

	def call(self, x: torch.Tensor) -> torch.Tensor:
		with torch.no_grad():
			y_hat = self.core_model(x)
		x_c = torch.concatenate((x, y_hat), dim=-1)
		y_hat_c = self.confidence_model(x_c)
		return y_hat_c

	def export_config(self) -> typing.Dict[str, typing.Any]:
		return self.args
