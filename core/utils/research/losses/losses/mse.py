import torch

from core.utils.research.losses import SpinozaLoss


class MeanSquaredErrorLoss(SpinozaLoss):

	def _call(self, y_hat: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
		return torch.mean((y_hat - y) ** 2, dim=1)
