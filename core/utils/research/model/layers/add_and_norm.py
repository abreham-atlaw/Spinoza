import typing

import torch
from torch import nn

from .add import Add
from ..model.savable import SpinozaModule


class AddAndNorm(SpinozaModule):

	def __init__(self, norm_layer, *args, **kwargs):
		self.args = {
			"norm_layer": norm_layer
		}
		super().__init__(*args, **kwargs)
		self.norm = norm_layer
		self.add = Add()

	def call(self, x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
		return self.norm(self.add(x, y))

	def export_config(self) -> typing.Dict[str, typing.Any]:
		return self.args
