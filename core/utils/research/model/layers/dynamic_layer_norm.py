import typing

import torch
from torch import nn

from core.utils.research.model.model.savable import SpinozaModule


class DynamicLayerNorm(SpinozaModule):

	def __init__(self, *args, elementwise_affine: bool = True, **kwargs):
		self.args = {
			"elementwise_affine": elementwise_affine
		}
		super().__init__(*args, **kwargs)
		self.norm_layer = None
		self.elementwise_affine = elementwise_affine

	def norm(self, x: torch.Tensor):
		if self.norm_layer is None:
			self.norm_layer = nn.LayerNorm(x.size()[-1:], elementwise_affine=self.elementwise_affine)
		return self.norm_layer(x)

	def call(self, x: torch.Tensor) -> torch.Tensor:
		return self.norm(x)

	def export_config(self) -> typing.Dict[str, typing.Any]:
		return self.args
