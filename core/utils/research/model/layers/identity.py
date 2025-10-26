import typing

import torch
import torch.nn as nn

from core.utils.research.model.model.savable import SpinozaModule


class Identity(SpinozaModule, nn.Identity):

	def call(self, x: torch.Tensor) -> torch.Tensor:
		return x

	def export_config(self) -> typing.Dict[str, typing.Any]:
		return {}
