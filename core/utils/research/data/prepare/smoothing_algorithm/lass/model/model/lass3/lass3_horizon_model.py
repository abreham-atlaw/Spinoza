import torch

from core.utils.research.model.model.savable import SpinozaModule
from core.utils.research.model.model.utils import AbstractHorizonModel


class Lass3HorizonModel(AbstractHorizonModel):

	def _check_depth(self, depth: int, x: torch.Tensor) -> bool:
		return super()._check_depth(depth, x) and torch.all(x[:, 1, -1] != 0)

	def _shift(self, x: torch.Tensor) -> torch.Tensor:
		x[:, 1, 1:] = x[:, 1, :-1].clone()
		return x

	def _place_prediction(self, x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
		x[:, 1, -1:] = y
		return x
