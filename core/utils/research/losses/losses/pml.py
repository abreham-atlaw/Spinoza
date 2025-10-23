import typing

import numpy as np
import torch
from torch import nn

from core.utils.research.losses import SpinozaLoss


class ProximalMaskedLoss(SpinozaLoss):

	def __init__(
			self,
			n: int,
			*args,
			p: int = 1,
			e: float = 1,
			r: float = 1,
			weights: typing.Optional[typing.Union[torch.Tensor, typing.List[float], np.ndarray]] = None,
			softmax=True,
			epsilon=1e-9,
			device='cpu',
			**kwargs
	):
		super().__init__(*args, **kwargs)
		self.n = n
		self.p = p
		self.e = e
		self.r = r
		self.activation = nn.Softmax(dim=1) if softmax else nn.Identity()
		self.mask = self._generate_mask().to(device)
		self.epsilon = epsilon
		self.device = device

		if weights is None:
			weights = torch.ones(n)
		if isinstance(weights, list):
			weights = torch.Tensor(weights)
		if isinstance(weights, np.ndarray):
			weights = torch.from_numpy(weights)

		self.w = weights.to(device)

	def _f(self, i: int) -> torch.Tensor:
		return (1 / (torch.abs(torch.arange(self.n) - i) + 1)) ** self.p

	def _generate_mask(self) -> torch.Tensor:

		return torch.stack([
			self._f(t) for t in range(self.n)
		])

	def collapse(self, loss: torch.Tensor) -> torch.Tensor:
		return torch.mean(loss)

	def _loss(self, y_mask: torch.Tensor, y_hat: torch.Tensor) -> torch.Tensor:
		return (1 / (torch.sum(y_mask * y_hat, dim=1) - self.epsilon)) - 1

	def _call(self, y_hat: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
		y_hat = self.activation(y_hat)
		y_hat = y_hat**self.e

		y_mask = torch.sum(
			self.mask * torch.unsqueeze(y, dim=2),
			dim=1
		)

		loss = self._loss(y_mask, y_hat)**(1/self.r)
		w = torch.sum(self.w * y, dim=1)
		loss = loss*w

		return loss
