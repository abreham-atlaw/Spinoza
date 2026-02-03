import typing

import torch
from torch import nn

from core.utils.research.model.layers import FlattenLayer
from core.utils.research.model.model.linear.model import LinearModel
from core.utils.research.model.model.savable import SpinozaModule
from .dense_confidence_input_block import DenseConfidenceInputBlock


class DenseConfidenceModel(SpinozaModule):

	def __init__(
			self,
			input_shape: typing.Tuple[int, int],
			input_block: DenseConfidenceInputBlock,
			encoder: LinearModel,
			head: LinearModel,
	):
		self.args = {
			"input_shape": input_shape,
			"input_block": input_block,
			"encoder": encoder,
			"head": head,
		}
		super().__init__(input_size=(None,)+tuple(input_shape), auto_build=False)
		self.input_block = input_block if input_block is not None else nn.Identity()
		self.encoder = encoder
		self.head = head
		self.flatten_layer = FlattenLayer(1, 2)
		self.init()

	def call(self, x: torch.Tensor) -> torch.Tensor:
		x = self.input_block(x)
		encoder_y = self.encoder(x)
		flatten = self.flatten_layer(encoder_y)
		y = self.head(flatten)
		return y

	def export_config(self) -> typing.Dict[str, typing.Any]:
		return self.args
