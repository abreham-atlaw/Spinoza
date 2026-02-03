import typing

import torch

from core.utils.research.model.layers import Identity
from core.utils.research.model.model.savable import SpinozaModule


class DenseConfidenceInputBlock(SpinozaModule):

	def __init__(
			self,
			split_point: int,
			left_norm: SpinozaModule,
			right_norm: SpinozaModule,
	):
		self.args = {
			"split_point": split_point,
			"left_norm": left_norm,
			"right_norm": right_norm,
		}
		super().__init__()
		self.split_point = split_point
		self.left_norm = left_norm if left_norm is not None else Identity()
		self.right_norm = right_norm if right_norm is not None else Identity()

	def call(self, x: torch.Tensor) -> torch.Tensor:
		left, right = x[..., :self.split_point], x[..., self.split_point:]
		left = self.left_norm(left)
		right = self.right_norm(right)
		y = torch.concatenate((left, right), dim=-1)
		return y

	def export_config(self) -> typing.Dict[str, typing.Any]:
		return self.args
