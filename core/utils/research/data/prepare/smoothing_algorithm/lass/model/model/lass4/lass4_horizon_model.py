import torch

from core.utils.research.data.prepare.smoothing_algorithm.lass.model.model.lass3 import Lass3HorizonModel


class Lass4HorizonModel(Lass3HorizonModel):

	def _place_prediction(self, x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
		x[:, 1, -1:] = y + x[:, 1, -2:-1]
		return x
