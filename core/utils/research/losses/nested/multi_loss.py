import typing

import torch
import torch.nn as nn
import numpy as np

from core.utils.research.losses import MeanSquaredClassError, SpinozaLoss


class MultiLoss(SpinozaLoss):

	def __init__(
			self,
			losses: typing.List[nn.Module],
			weights: typing.Union[typing.List[float], np.ndarray, torch.Tensor] = None,
			*args,
			device: torch.device = None,
			**kwargs
	):
		super().__init__(*args, **kwargs)
		self.losses = nn.ModuleList(losses)
		if weights is None:
			weights = np.ones(len(losses))
		if isinstance(weights, list):
			weights = torch.Tensor(weights)
		if isinstance(weights, np.ndarray):
			weights = torch.from_numpy(weights)
		
		if device is None:
			for loss in self.losses:
				if hasattr(loss, "device"):
					device = loss.device
					break
		if device is None:
			device = torch.device("cpu")
		self.device = device

		self.weights = weights.to(self.device)

	def _call(self, y_hat: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
		return torch.sum(
			torch.stack([
				loss(y_hat, y)
				for loss in self.losses
			], dim=1) * self.weights,
			dim=1
		)

	def __str__(self):
		return "{}[{}]".format(super().__str__(), ', '.join([str(loss) for loss in self.losses]))


class MSCECrossEntropyLoss(MultiLoss):

	def __init__(self, classes: typing.Union[np.ndarray, torch.Tensor], epsilon: float = None, device=None, *args,
				 **kwargs):
		super().__init__(
			[MeanSquaredClassError(classes, epsilon, device), nn.CrossEntropyLoss()],
			*args,
			device=device,
			**kwargs
		)
